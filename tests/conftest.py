"""Round-2 gate hooks: random test order (TEST_SEED) and xdist workers
(TEST_WORKERS); see the programme's tools/round2_pytest.py for the contract.

TEST_SEED unset = a fresh seed each run, printed first as TEST_SEED=<n>;
TEST_SEED=<n> replays it; TEST_SEED=off keeps file order. TEST_WORKERS=1 runs
serially without xdist; unset = all cores up to WORKER_CAP.
"""
from __future__ import annotations

import importlib
import os
import random
import types

import pytest

WORKER_CAP = 12  # tune per repo: more workers than this only pays startup


def _seed() -> int | None:
    raw = os.environ.get("TEST_SEED", "").strip().lower()
    if raw == "off":
        return None
    if not raw:
        raw = str(random.SystemRandom().randrange(1, 2**31))
        os.environ["TEST_SEED"] = raw  # xdist workers inherit it
    return int(raw)


@pytest.hookimpl(tryfirst=True)
def pytest_cmdline_main(config):
    if os.environ.get("PYTEST_XDIST_WORKER"):
        return
    _seed()
    try:
        import xdist  # noqa: F401
    except ImportError:
        xdist = None
    if xdist and getattr(config.option, "numprocesses", None) is None:
        raw = os.environ.get("TEST_WORKERS", "").strip()
        n = int(raw) if raw else min(os.cpu_count() or 1, WORKER_CAP)
        if n > 1:
            config.option.numprocesses = n
            if getattr(config.option, "dist", "no") == "no":
                config.option.dist = "load"
    workers = getattr(config.option, "numprocesses", None) or 1
    # Printed before capture starts, so it shows even under -q.
    print(f"TEST_SEED={os.environ.get('TEST_SEED', 'off')} TEST_WORKERS={workers}", flush=True)


def pytest_collection_modifyitems(session, config, items):
    seed = _seed()
    if seed is None:
        return
    rng = random.Random(seed)
    files: dict[str, list] = {}
    for item in items:
        files.setdefault(item.nodeid.split("::", 1)[0], []).append(item)
    groups = list(files.values())
    rng.shuffle(groups)
    for group in groups:
        rng.shuffle(group)
    items[:] = [item for group in groups for item in group]


# ─── one registry, one store, for every module ────────────────────────────
# src.indexer.store and the src.tools.* modules do `from src.config import
# VAULTS` (and temporal/librarian `from src.indexer.store import _entities,
# _observations`), binding whichever dict was current when they were first
# imported. Production creates each of these once and only ever mutates it in
# place (_load_vaults at import, create/delete_vault, _load_store). Tests
# instead rebind them in setUp (`config_mod.VAULTS = {...}`,
# `store_mod._entities = {}`), so which dict a tool module saw depended on
# which test imported it first -- invisible in file order, exposed by random
# order (test_maintenance counting another file's entity as an orphan vault,
# test_portability's point-in-time read finding no observations, test_eval's
# fixture vault missing from the store's registry). Here such a rebind
# refills the one shared dict instead, which is what production does.
_SHARED_DICTS = {
    "src.config": ("VAULTS",),
    "src.indexer.store": ("_entities", "_observations"),
}


class _RefillInPlace(types.ModuleType):
    def __setattr__(self, name, value):
        if name in _SHARED_DICTS.get(self.__name__, ()) and isinstance(value, dict):
            shared = self.__dict__.get(name)
            if isinstance(shared, dict) and shared is not value:
                shared.clear()
                shared.update(value)
                return
        super().__setattr__(name, value)


def _share_dicts() -> None:
    for name in _SHARED_DICTS:
        module = importlib.import_module(name)
        if not isinstance(module, _RefillInPlace):
            module.__class__ = _RefillInPlace


_share_dicts()
