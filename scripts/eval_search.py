"""Retrieval eval harness for memory-index search ("the answer key").

Runs a fixed set of golden queries through the real `search_memory` code path
and reports recall@k and MRR, plus a per-query pass/fail table.

Two modes
---------
1. Fixture mode (default, self-contained)
     uv run python scripts/eval_search.py

   Builds the synthetic vault from tests/eval_golden.json into a throwaway
   temp data dir through the *real* store APIs (create_entity /
   add_observations -> real ChromaDB collection), but swaps the embedder for a
   deterministic hash bag-of-words stand-in. The real EmbeddingGemma ONNX model
   is not required, and the numbers are byte-for-byte reproducible.

   IMPORTANT: fixture-mode numbers validate PLUMBING, not model quality. They
   tell you that ranking order, the calibrated threshold gate, the min-3
   fallback, the JSON contract and the metric code all behave. They say
   nothing about how well EmbeddingGemma understands a paraphrase — a
   lexical-overlap embedder is not a semantic one.

2. Live mode
     uv run python scripts/eval_search.py --data-dir /srv/memory-index/data

   Points every path at a real data dir and uses the real embedder + real
   Chroma. By default it ingests the same synthetic corpus into the vault named
   in the golden file, so you get a real-model quality number on a corpus whose
   answers are known. Add --no-ingest --vault work --golden <your.json> to
   evaluate an existing vault against its own answer key instead.

     uv run python scripts/eval_search.py --data-dir ./data --no-ingest \\
         --vault work --golden tests/work_golden.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_GOLDEN = PROJECT_ROOT / "tests" / "eval_golden.json"
DEFAULT_K = 5


# --------------------------------------------------------------------------
# Deterministic stand-in embedder
# --------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Function words carry no retrieval signal and would otherwise dominate the
# cosine between two short sentences.
_STOPWORDS = frozenset("""
a an the and or of to in on at for from with by as is are was were be been being
it its this that these those any some no not do does did done have has had
you your we our they their he she his her i me my
""".split())


class DeterministicEmbedder:
    """Hash bag-of-words -> L2-normalized vector. Same text, same vector, always.

    Deliberately *lexical*: overlapping words pull vectors together, unrelated
    text sits near-orthogonal. That is enough to exercise every ranking,
    threshold and formatting path in search_memory with stable, reproducible
    distances — and it is emphatically not a semantic model.

    Vectors are L2-normalized like EmbeddingGemma's, so Chroma's default
    squared-L2 distance lands in [0, 4] (d = 2 - 2*cos) and the calibrated
    default thresholds (HIGH 0.6 / MEDIUM 1.0 / LOW 1.4) remain meaningful.

    Mirrors the GemmaEmbedder interface used by the store (`__call__` for
    documents) and by search (`embed_queries` for queries).
    """

    dim = 512
    backend = "deterministic hash-BoW (fixture)"

    def _slot(self, token: str) -> int:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big") % self.dim

    def _vec(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in _TOKEN_RE.findall((text or "").lower()):
            if token in _STOPWORDS:
                continue
            vec[self._slot(token)] += 1.0
            if len(token) >= 4:
                # A crude stem slot so "sign"/"signs"/"signed" partially agree.
                vec[self._slot(token[:4] + "~")] += 0.5
        norm = sum(v * v for v in vec) ** 0.5
        if norm == 0.0:
            # Empty / all-stopword text: a fixed unit vector, so it is
            # equidistant from everything rather than a division by zero.
            vec[0] = 1.0
            return vec
        return [v / norm for v in vec]

    def __call__(self, input):  # noqa: A002 - Chroma's parameter name
        return [self._vec(t) for t in input]

    def embed_queries(self, queries: list[str]) -> list[list[float]]:
        return [self._vec(q) for q in queries]

    def close(self) -> None:
        pass


# --------------------------------------------------------------------------
# Environment wiring
# --------------------------------------------------------------------------

def _shutdown_chroma(client, clear_cache: bool) -> None:
    """Release a PersistentClient's sqlite handles.

    Chroma has no public close(); on Windows the open sqlite file keeps the
    whole data dir undeletable, so a fixture run would leak a temp directory
    every time. Stopping the System closes the DB component.
    """
    if client is None:
        return
    try:
        client._system.stop()
    except Exception:
        pass
    if clear_cache:
        # Chroma caches Systems by settings; without this a later client for the
        # same path is handed the stopped one.
        try:
            from chromadb.api.shared_system_client import SharedSystemClient
            SharedSystemClient.clear_system_cache()
        except Exception:
            pass


@contextmanager
def eval_environment(data_dir: Path, mock_embedder: bool = True,
                     disable_write_hooks: bool = True):
    """Repoint every module-level path at `data_dir`, then restore on exit.

    The modules bind DATA_DIR & friends at import time, so each one has to be
    rebound individually — the same approach tests/test_store.py takes, done by
    hand here so the script can also run against a real data dir.
    """
    import src.config as config
    import src.graph.manager as graph_manager
    import src.indexer.calibration as calibration
    import src.indexer.embedder as embedder
    import src.indexer.store as store
    import src.tools.search as search

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    saved = {
        "config": {k: getattr(config, k) for k in
                   ("DATA_DIR", "CHROMA_DIR", "VAULTS_FILE", "ENTITIES_FILE",
                    "GRAPH_FILE")},
        "vaults": dict(config.VAULTS),
        "embedder": {k: getattr(embedder, k) for k in
                     ("CHROMA_DIR", "_client", "_embedding_fn", "_active_backend")},
        "calibration_data_dir": calibration.DATA_DIR,
        "store": {k: getattr(store, k) for k in
                  ("DATA_DIR", "ENTITIES_FILE", "_entities", "_observations",
                   "_loaded", "_run_post_write_hooks")},
        "graph": {k: getattr(graph_manager, k) for k in
                  ("DATA_DIR", "GRAPH_FILE", "_graph", "_relations")},
        "calibration_cache": dict(search._calibration_cache),
    }

    try:
        config.DATA_DIR = data_dir
        config.CHROMA_DIR = data_dir / "chroma"
        config.VAULTS_FILE = data_dir / "vaults.json"
        config.ENTITIES_FILE = data_dir / "memory_entities.json"
        config.GRAPH_FILE = data_dir / "memory_graph.json"

        store.DATA_DIR = data_dir
        store.ENTITIES_FILE = config.ENTITIES_FILE
        store._entities = {}
        store._observations = {}
        store._loaded = False  # force a reload from the new ENTITIES_FILE
        if disable_write_hooks:
            # Auto-recalibration and the auto-librarian fire every 10
            # observations. Both would mutate thresholds mid-ingest (and spawn
            # background threads), which is exactly what an eval must not have.
            store._run_post_write_hooks = lambda *a, **kw: None

        calibration.DATA_DIR = data_dir

        graph_manager.DATA_DIR = data_dir
        graph_manager.GRAPH_FILE = config.GRAPH_FILE
        graph_manager._graph = None
        graph_manager._relations = {}

        embedder.CHROMA_DIR = config.CHROMA_DIR
        embedder._client = None
        if mock_embedder:
            embedder._embedding_fn = DeterministicEmbedder()
            embedder._active_backend = DeterministicEmbedder.backend
        else:
            embedder._embedding_fn = None
            embedder._active_backend = "not initialized"

        # VAULTS is imported by reference elsewhere — mutate, never rebind.
        config.VAULTS.clear()
        if config.VAULTS_FILE.exists():
            try:
                raw = json.loads(config.VAULTS_FILE.read_text(encoding="utf-8"))
                for name, cfg in (raw.get("vaults") or {}).items():
                    config.VAULTS[name] = config.VaultConfig.from_dict(cfg)
            except (json.JSONDecodeError, OSError):
                pass

        search._calibration_cache.clear()

        yield

    finally:
        # Close whatever client this run opened, but never the one that was
        # already there (it belongs to the surrounding process).
        _shutdown_chroma(embedder._client,
                         clear_cache=saved["embedder"]["_client"] is None)
        for key, value in saved["config"].items():
            setattr(config, key, value)
        config.VAULTS.clear()
        config.VAULTS.update(saved["vaults"])
        for key, value in saved["embedder"].items():
            setattr(embedder, key, value)
        calibration.DATA_DIR = saved["calibration_data_dir"]
        for key, value in saved["store"].items():
            setattr(store, key, value)
        for key, value in saved["graph"].items():
            setattr(graph_manager, key, value)
        search._calibration_cache.clear()
        search._calibration_cache.update(saved["calibration_cache"])


def load_golden(path: Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not data.get("queries"):
        raise ValueError(f"{path} has no queries")
    return data


def ingest_fixture(golden: dict, vault: str) -> tuple[int, int]:
    """Build the fixture vault through the real store APIs.

    Returns (entity_count, observation_count).
    """
    from src.config import VAULTS, VaultConfig, create_vault
    from src.indexer.store import create_entity

    if vault not in VAULTS:
        try:
            create_vault(vault)
        except OSError:
            VAULTS[vault] = VaultConfig(name=vault)

    n_obs = 0
    for spec in golden.get("entities", []):
        observations = list(spec.get("observations", []))
        create_entity(
            spec["name"], spec.get("type", "concept"), vault,
            observations=observations, source=spec.get("source", ""),
        )
        n_obs += len(observations)
    return len(golden.get("entities", [])), n_obs


def calibrate(vault: str, seed: int = 1234) -> dict:
    """Derive per-vault thresholds from the vault's own distance distribution.

    calibrate_collection samples probes with `random`, so the RNG is seeded to
    keep the eval reproducible. Without this the harness uses the default
    priors (HIGH 0.6 / MEDIUM 1.0 / LOW 1.4), which is the right baseline for a
    vault that has never been calibrated.
    """
    import random

    from src.indexer.calibration import calibrate_collection
    from src.indexer.store import _get_collection_for_vault
    from src.tools.search import invalidate_calibration_cache

    random.seed(seed)
    result = calibrate_collection(_get_collection_for_vault(vault), vault)
    invalidate_calibration_cache(vault)
    return result["thresholds"]


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def _is_relevant(result: dict, expect_contents: set[str],
                 expect_entities: set[str]) -> bool:
    if expect_contents and (result.get("content") or "").strip() in expect_contents:
        return True
    if expect_entities and (result.get("entity_name") or "") in expect_entities:
        return True
    return False


def score_query(entry: dict, payload: dict, k: int) -> dict:
    """recall@k and reciprocal rank for one golden query.

    recall@k = (distinct expected items found in the top k) / (expected items).
    An entity-level entry (no expect_contents) needs exactly one hit on that
    entity, so its recall is 0.0 or 1.0.
    """
    expect_contents = {c.strip() for c in entry.get("expect_contents", [])}
    expect_entities = set(entry.get("expect_entities", []))
    if not expect_contents and not expect_entities:
        raise ValueError(f"golden query {entry.get('id')} expects nothing")

    results = payload.get("results", []) or []
    total_relevant = len(expect_contents) if expect_contents else 1
    top = results[:k]

    found: set[str] = set()
    first_rank = 0
    for rank, result in enumerate(top, start=1):
        if not _is_relevant(result, expect_contents, expect_entities):
            continue
        if first_rank == 0:
            first_rank = rank
        found.add((result.get("content") or "").strip()
                  if expect_contents else result.get("entity_name", ""))

    hits = min(len(found), total_relevant)
    return {
        "id": entry.get("id", ""),
        "query": entry.get("query", ""),
        "expected": total_relevant,
        "hits": hits,
        "recall": hits / total_relevant if total_relevant else 0.0,
        "rr": 1.0 / first_rank if first_rank else 0.0,
        "first_rank": first_rank,
        "returned": len(results),
        # How many results cleared the calibrated noise floor. Fewer than
        # MIN_RESULTS means search_memory's min-3 fallback filled the gap.
        "above_threshold": payload.get("above_threshold") or 0,
        "min3_fallback": (payload.get("above_threshold") or 0) < 3,
        "passed": hits == total_relevant,
        "top_hit": (top[0].get("content", "") if top else ""),
        "top_confidence": (top[0].get("confidence", "") if top else ""),
        "top_pct": (top[0].get("relevance_pct", 0.0) if top else 0.0),
    }


def run_queries(golden: dict, vault: str, k: int, strategy: str) -> list[dict]:
    from src.tools.search import search_memory

    rows = []
    for entry in golden["queries"]:
        raw = search_memory(
            entry["query"], vault=vault, n_results=k, strategy=strategy,
            output_format="json",
        )
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            # search_memory returns a plain error string on failure.
            payload = {"results": []}
            print(f"  ! {entry.get('id')}: {raw.strip()[:120]}")
        rows.append(score_query(entry, payload, k))
    return rows


def aggregate(rows: list[dict], k: int) -> dict:
    n = len(rows) or 1
    return {
        "queries": len(rows),
        "k": k,
        f"recall@{k}": sum(r["recall"] for r in rows) / n,
        "mrr": sum(r["rr"] for r in rows) / n,
        "passed": sum(1 for r in rows if r["passed"]),
        "failed": sum(1 for r in rows if not r["passed"]),
        "mean_returned": sum(r["returned"] for r in rows) / n,
        "mean_above_threshold": sum(r["above_threshold"] for r in rows) / n,
        "min3_fallback_queries": sum(1 for r in rows if r["min3_fallback"]),
    }


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def _truncate(text: str, width: int) -> str:
    """ASCII-only truncation — this prints to a cp1252 console on Windows."""
    text = (text or "").replace("\n", " ")
    return text if len(text) <= width else text[: width - 3] + "..."


def print_report(rows: list[dict], agg: dict, header: dict) -> None:
    k = agg["k"]
    print()
    for key, value in header.items():
        print(f"{key:>12}: {value}")
    print()
    print(f"{'id':<5} {'ok':<5} {'n':<3} {'>thr':<5} {'rank':<5} {'rec':<5} "
          f"{'rr':<5} {'query':<44} top hit")
    print("-" * 122)
    for row in rows:
        mark = "PASS" if row["passed"] else "FAIL"
        rank = str(row["first_rank"]) if row["first_rank"] else "-"
        print(f"{row['id']:<5} {mark:<5} {row['returned']:<3} "
              f"{row['above_threshold']:<5} {rank:<5} "
              f"{row['recall']:<5.2f} {row['rr']:<5.2f} "
              f"{_truncate(row['query'], 44):<44} "
              f"{_truncate(row['top_hit'], 40)}")
    print("-" * 122)
    print(f"queries          : {agg['queries']}")
    print(f"recall@{k}         : {agg[f'recall@{k}']:.3f}")
    print(f"MRR              : {agg['mrr']:.3f}")
    print(f"pass / fail      : {agg['passed']} / {agg['failed']}")
    print(f"mean returned    : {agg['mean_returned']:.2f} of k={k}")
    print(f"mean above thr.  : {agg['mean_above_threshold']:.2f} "
          f"(min-3 fallback engaged on {agg['min3_fallback_queries']} "
          f"quer{'y' if agg['min3_fallback_queries'] == 1 else 'ies'})")

    failures = [r for r in rows if not r["passed"]]
    if failures:
        print("\nmisses:")
        for row in failures:
            print(f"  {row['id']} -> {_truncate(row['query'], 60)}")
            print(f"        got {row['returned']} result(s), "
                  f"{row['above_threshold']} above threshold; "
                  f"top: {_truncate(row['top_hit'], 60)}")


# --------------------------------------------------------------------------
# Entry points
# --------------------------------------------------------------------------

def run_eval(golden_path: Path = DEFAULT_GOLDEN, data_dir: Path | None = None,
             vault: str = "", k: int = DEFAULT_K, strategy: str = "semantic",
             ingest: bool = True, recalibrate: bool = False) -> dict:
    """Run the full eval. Returns {"rows": [...], "aggregate": {...}, ...}.

    data_dir=None  -> fixture mode: temp dir + deterministic mock embedder.
    data_dir=<p>   -> live mode: real paths, real embedder, real model.
    """
    golden = load_golden(Path(golden_path))
    vault = vault or golden.get("vault", "eval_fixture")
    fixture_mode = data_dir is None

    tmpdir = tempfile.mkdtemp(prefix="memidx-eval-") if fixture_mode else None
    target_dir = Path(tmpdir) if fixture_mode else Path(data_dir)

    try:
        with eval_environment(target_dir, mock_embedder=fixture_mode):
            n_entities = n_obs = 0
            if ingest:
                n_entities, n_obs = ingest_fixture(golden, vault)

            from src.config import VAULTS
            if vault not in VAULTS:
                raise SystemExit(
                    f"Vault '{vault}' does not exist in {target_dir}. "
                    f"Known vaults: {sorted(VAULTS) or 'none'}"
                )

            if recalibrate:
                thresholds = calibrate(vault)
            else:
                from src.indexer.calibration import get_thresholds
                thresholds = get_thresholds(vault)

            rows = run_queries(golden, vault, k, strategy)
            agg = aggregate(rows, k)

            return {
                "rows": rows,
                "aggregate": agg,
                "mode": "fixture" if fixture_mode else "live",
                "vault": vault,
                "data_dir": str(target_dir),
                "ingested_entities": n_entities,
                "ingested_observations": n_obs,
                "thresholds": thresholds,
                "strategy": strategy,
                "calibrated": recalibrate,
            }
    finally:
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Retrieval eval for memory-index search_memory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data-dir", default=None,
        help="Live mode: run against this data dir with the REAL embedder. "
             "Omit for self-contained fixture mode (temp dir + mock embedder).",
    )
    parser.add_argument("--golden", default=str(DEFAULT_GOLDEN),
                        help="Golden dataset JSON (default: tests/eval_golden.json)")
    parser.add_argument("--vault", default="",
                        help="Vault to search (default: the golden file's vault)")
    parser.add_argument("-k", "--k", type=int, default=DEFAULT_K,
                        help=f"Cutoff for recall@k (default {DEFAULT_K})")
    parser.add_argument("--strategy", default="semantic",
                        choices=["semantic", "associative"])
    parser.add_argument("--no-ingest", action="store_true",
                        help="Do not write the fixture corpus; evaluate the "
                             "vault exactly as it is on disk (live mode only)")
    parser.add_argument("--calibrate", action="store_true",
                        help="Recalibrate the vault's confidence bands from its "
                             "own distance distribution before evaluating "
                             "(seeded, reproducible). Default: use whatever "
                             "calibration is on disk, else the default priors.")
    parser.add_argument("--min-recall", type=float, default=0.0,
                        help="Exit non-zero if mean recall@k falls below this")
    parser.add_argument("--json", action="store_true",
                        help="Emit the raw result payload as JSON instead of a table")
    args = parser.parse_args(argv)

    if args.no_ingest and args.data_dir is None:
        parser.error("--no-ingest only makes sense with --data-dir "
                     "(fixture mode has nothing to read).")

    report = run_eval(
        golden_path=Path(args.golden),
        data_dir=Path(args.data_dir) if args.data_dir else None,
        vault=args.vault, k=args.k, strategy=args.strategy,
        ingest=not args.no_ingest, recalibrate=args.calibrate,
    )

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        header = {
            "mode": report["mode"],
            "vault": report["vault"],
            "data dir": report["data_dir"],
            "corpus": f"{report['ingested_entities']} entities / "
                      f"{report['ingested_observations']} observations ingested",
            "strategy": report["strategy"],
            "bands": ", ".join(f"{k}<{v}" for k, v in report["thresholds"].items())
                     + (" (recalibrated)" if report["calibrated"] else " (on disk/default)"),
        }
        print_report(report["rows"], report["aggregate"], header)
        if report["mode"] == "fixture":
            print("\nNOTE: fixture mode uses a deterministic lexical stand-in for "
                  "EmbeddingGemma.\n      These numbers validate PLUMBING "
                  "(ranking order, threshold gate, min-3\n      fallback, JSON "
                  "contract, metric code) -- NOT real-model retrieval quality.\n"
                  "      Run with --data-dir on the server for a quality number.")

    recall = report["aggregate"][f"recall@{args.k}"]
    if recall < args.min_recall:
        print(f"\nFAILED: recall@{args.k} {recall:.3f} < "
              f"min-recall {args.min_recall:.3f}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
