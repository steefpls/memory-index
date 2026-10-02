"""The `private` vault: only the box's owner may use it.

Each box has a `private` vault for its owner's personal life, health, money,
family, private opinions of people and anything from chats the owner keeps
private. Every tool refuses it -- search, read, write, list, export, graph,
all of them -- to a call that isn't made for the owner: a chat turn for the
other owner, for the peer assistant or for a trusted non-owner, and any job
or wake one of those started. Owner turns, the owner's own sessions and
owner-started hub work (programmes, schedules, the gardener) may use it.
(Steve's decision, 2026-10-02.)

Who a call is for is decided once per HTTP request, from the process at the
other end of the connection and the hub's records of the run it belongs to
(src/requester.py has the rules). It is never taken from anything the model
writes, and anything that can't be vouched for as the owner's is refused.

How each kind of refusal looks to the caller:

  - A tool aimed at the vault by name (vault="private", create_vault,
    export_vault, ...) is refused with a plain message.
  - Everything else behaves as if the vault weren't there: all-vault
    searches, lists, timelines and graph reads silently leave it out; an
    entity, observation or relation in it is "not found", by name or by ID.

The checks sit in the store and graph read paths (indexer/store.py,
graph/manager.py), so a tool can't reach a private row by a path nobody
thought to guard, and in the tools that take a vault name.

Calls with no HTTP request behind them -- stdio mode, tests, the daemon's
own startup and background work -- are the owner's, EXCEPT inside the HTTP
daemon, where a call that somehow lost its request is refused (`serving`).
Background threads the daemon starts on purpose run under `system()`.
"""
from __future__ import annotations

import contextlib
import contextvars
import functools
import logging
import threading
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Mapping

logger = logging.getLogger(__name__)

PRIVATE_VAULT = "private"

REFUSAL = ("The private vault is only for {owner}'s own conversations, so this "
           "call can't use it. Nothing was read or written there. Don't write a "
           "private fact to another vault instead: in this conversation it isn't "
           "written down anywhere.")

# Tools that reach rows a non-owner shouldn't see whatever vault they name:
# an export holds deleted rows (the notes moved out of `work` are among
# them), an import loads any archive on disk, undelete revives any id
# (review A5/B2, 2026-10-02).
OWNER_ONLY = ("That tool is only for {owner}'s own conversations: it can reach notes "
              "this conversation isn't allowed to see. Nothing was done.")


def is_private(vault: str | None) -> bool:
    """Is `vault` the protected one? Case and spacing don't make a second."""
    return (vault or "").strip().lower() == PRIVATE_VAULT


@dataclass(frozen=True)
class Verdict:
    """Who a call is for: the owner (may use private) or not, and why."""
    owner: bool
    why: str


OWNER_DIRECT = Verdict(True, "no HTTP request (stdio, tests or the daemon itself)")
LOST_REQUEST = Verdict(False, "a call in the HTTP daemon with no request behind it")
SYSTEM = Verdict(True, "the daemon's own background work")


class _Request:
    """One HTTP request's caller, decided on first use and then kept."""

    def __init__(self, peer: tuple[str, int] | None, headers: Mapping[str, str]) -> None:
        self.peer = peer
        self.headers = dict(headers)
        self._verdict: Verdict | None = None
        self._lock = threading.Lock()

    def verdict(self) -> Verdict:
        with self._lock:
            if self._verdict is None:
                try:
                    from src import requester
                    self._verdict = requester.decide(self.peer, self.headers)
                except Exception as e:  # noqa: BLE001 -- fail closed
                    logger.warning("private vault: couldn't decide who %s is for: %r", self.peer, e)
                    self._verdict = Verdict(False, f"the lookup failed ({type(e).__name__})")
                if not self._verdict.owner:
                    logger.info("private vault hidden from %s: %s", self.peer, self._verdict.why)
            return self._verdict


_request: contextvars.ContextVar[_Request | None] = contextvars.ContextVar(
    "memory_index_request", default=None)
_forced: contextvars.ContextVar[Verdict | None] = contextvars.ContextVar(
    "memory_index_forced", default=None)

# Set once the HTTP daemon starts serving: from then on a tool call that has
# no request in its context is refused rather than read as the owner's.
_serving = False


def serving(on: bool = True) -> None:
    global _serving
    _serving = on


def verdict() -> Verdict:
    """Who the current call is for."""
    forced = _forced.get()
    if forced is not None:
        return forced
    req = _request.get()
    if req is not None:
        return req.verdict()
    return LOST_REQUEST if _serving else OWNER_DIRECT


def private_allowed() -> bool:
    return verdict().owner


def hidden_vaults() -> frozenset[str]:
    """Vault names this call may not see (normalised: compare with is_hidden)."""
    return frozenset() if private_allowed() else frozenset({PRIVATE_VAULT})


def is_hidden(vault: str | None) -> bool:
    """May this call NOT see `vault`?"""
    return is_private(vault) and not private_allowed()


def refusal() -> str:
    from src import requester
    return REFUSAL.format(owner=requester.owner_name())


def refuse_unless_owner() -> str | None:
    """The refusal for an owner-only tool, or None for the owner."""
    if private_allowed():
        return None
    from src import requester
    return OWNER_ONLY.format(owner=requester.owner_name())


def refuse_vault(vault: str | None) -> str | None:
    """The refusal for a tool aimed at `vault` by name, or None when allowed."""
    return refusal() if is_hidden(vault) else None


@contextlib.contextmanager
def as_verdict(v: Verdict) -> Iterator[None]:
    """Run the block as `v` (tests; the daemon's own work via `system`)."""
    token = _forced.set(v)
    try:
        yield
    finally:
        _forced.reset(token)


def system() -> contextlib.AbstractContextManager[None]:
    """The daemon's own work (startup, re-embeds, the auto-librarian)."""
    return as_verdict(SYSTEM)


def in_system(fn: Callable[..., Any]) -> Callable[..., Any]:
    """`fn` wrapped to run under `system()`, for a thread's target: a new
    thread starts with no request, which the HTTP daemon would refuse."""
    @functools.wraps(fn)
    def run(*args: Any, **kwargs: Any) -> Any:
        with system():
            return fn(*args, **kwargs)
    return run


# --- the HTTP side ---------------------------------------------------------

def _headers(scope: Mapping[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in scope.get("headers") or ():
        try:
            out[k.decode("latin-1").lower()] = v.decode("latin-1")
        except Exception:  # noqa: BLE001
            continue
    return out


class CallerMiddleware:
    """Puts each HTTP request's caller in the context its tool calls run in.

    The decision itself is lazy: a request that never touches a vault (the
    /inflight poll, list_tools) costs nothing.
    """

    def __init__(self, app: Callable[..., Any]) -> None:
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        client = scope.get("client")
        peer: tuple[str, int] | None = None
        if client:
            try:
                peer = (str(client[0]), int(client[1]))
            except (TypeError, ValueError, IndexError):
                peer = None
        token = _request.set(_Request(peer, _headers(scope)))
        try:
            await self.app(scope, receive, send)
        finally:
            _request.reset(token)
