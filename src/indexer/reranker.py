"""Second-pass relevance gate: a MiniLM-L6 cross-encoder, ONNX.

Cosine distance ranks one query's hits well but cannot say whether a hit is
relevant at all: EmbeddingGemma's scores bunch up, so the same number is an
answer for one query and noise for the next. Measured on 86 real queries
(2026-10-09, scratch/memeval): the calibrated bands put 1,260 noise hits and
466 useful ones in LOW, and a no-answer query ("my favourite food") still
returned ~470 tokens of LOW hits. A cross-encoder reads the query and the
fact together and gives a probability that does carry across queries.

Search keeps its own order and only drops candidates this model scores under
RERANK_MIN (see src/tools/search.py). Re-sorting by this model lost answers in
the same eval; gating alone kept them and emptied the no-answer queries.

Loading is explicit: the server's search warm-up calls `start()`, scripts call
`load()`. Until one has, `get()` is None and search behaves as it did before
the gate, which is also what happens when the model is missing or fails.
Device follows MEMORY_INDEX_EMBED_DEVICE, with the embedder's CUDA smoke test
and idle arena release (GTX 1070: ~80 ms for 30 candidates; i7-8550U CPU:
~1.4 s).
"""

import logging
import os
import shutil
import threading
from contextlib import nullcontext
from pathlib import Path

from src.config import (
    RERANK_MAX_TOKENS,
    RERANK_ONNX_DIR,
    RERANK_ONNX_FILES,
    RERANK_ONNX_REPO,
    RERANK_ONNX_REVISION,
)
from src.indexer.embedder import (
    ARENA_RELEASE_SECONDS,
    EMBED_DEVICE,
    _ArenaReleaser,
    _CUDA,
    _run_options,
    open_session,
    providers_for,
)

logger = logging.getLogger(__name__)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    try:
        return float(raw) if raw else default
    except ValueError:
        logger.warning("%s=%r is not a number; using %s", name, raw, default)
        return default


# "off" keeps search exactly as it was before the gate.
RERANK_MODE = (os.environ.get("MEMORY_INDEX_RERANK", "on").strip().lower() or "on")
# Candidates scored under this are dropped. 0.1 was tuned on 66 of the 86 eval
# queries and held on the other 20: junk returned 28% -> 21%, no-answer
# queries emptied 6/7, queries missing every must-have 5.1% -> 6.3% (one query).
RERANK_MIN = _env_float("MEMORY_INDEX_RERANK_MIN", 0.1)
# How many of the vector ranking's best candidates the gate reads.
RERANK_POOL = 30


def model_present(model_dir: Path = RERANK_ONNX_DIR) -> bool:
    return all((model_dir / local).is_file() and (model_dir / local).stat().st_size > 0
               for _, local in RERANK_ONNX_FILES)


def download(model_dir: Path = RERANK_ONNX_DIR) -> None:
    """Fetch the pinned ONNX export and tokenizer (~90 MB, ungated)."""
    from huggingface_hub import hf_hub_download

    model_dir.mkdir(parents=True, exist_ok=True)
    for repo_path, local in RERANK_ONNX_FILES:
        dest = model_dir / local
        if dest.is_file() and dest.stat().st_size > 0:
            continue
        cached = hf_hub_download(RERANK_ONNX_REPO, repo_path, revision=RERANK_ONNX_REVISION)
        tmp = dest.with_suffix(dest.suffix + ".part")
        shutil.copyfile(cached, tmp)
        os.replace(tmp, dest)
        logger.info("Reranker: downloaded %s (%d bytes)", local, dest.stat().st_size)


class CrossEncoder:
    """MiniLM-L6 cross-encoder: (query, text) pairs -> relevance in [0, 1]."""

    def __init__(self, model_dir: Path = RERANK_ONNX_DIR):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self._tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self._tok.enable_truncation(max_length=RERANK_MAX_TOKENS)
        self._tok.enable_padding()
        # A separate lock: tokenizers' padding/truncation state is per instance.
        self._tok_lock = threading.Lock()

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        cores = os.cpu_count() or 4
        sess_opts.intra_op_num_threads = max(2, int(cores * 0.5))
        sess_opts.inter_op_num_threads = 1

        path = str(model_dir / "model.onnx")
        providers, err = providers_for(EMBED_DEVICE, list(ort.get_available_providers()))

        def smoke(session):
            self._inputs = {i.name for i in session.get_inputs()}
            self._output = session.get_outputs()[0].name
            session.run([self._output], self._feed("warmup", ["warmup"]),
                        _run_options(session, shrink=True))

        session, cuda_err = open_session(path, sess_opts, providers, smoke, "Reranker")
        err = err or cuda_err
        self._inputs = {i.name for i in session.get_inputs()}
        self._output = session.get_outputs()[0].name
        self._session = session
        self._run_opts = _run_options(session)
        self._releaser = None
        release_opts = _run_options(session, shrink=True)
        if release_opts is not None and ARENA_RELEASE_SECONDS > 0:
            tiny = self._feed("release", ["release"])
            self._releaser = _ArenaReleaser(
                lambda: session.run([self._output], tiny, release_opts), ARENA_RELEASE_SECONDS)
        self.device = "cuda" if session.get_providers()[0] == _CUDA else "cpu"
        self.device_error = err
        logger.info("Reranker loaded: MiniLM-L6 ONNX + %s", self.device.upper())

    def _feed(self, query: str, texts: list[str]) -> dict:
        import numpy as np

        with self._tok_lock:
            enc = self._tok.encode_batch([(query, t) for t in texts])
        feed = {
            "input_ids": np.array([e.ids for e in enc], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in enc], dtype=np.int64),
            "token_type_ids": np.array([e.type_ids for e in enc], dtype=np.int64),
        }
        return {k: v for k, v in feed.items() if k in self._inputs}

    def score(self, query: str, texts: list[str]) -> list[float]:
        import numpy as np

        if not texts:
            return []
        feed = self._feed(query, texts)
        releaser = self._releaser
        with releaser.lock if releaser is not None else nullcontext():
            logits = self._session.run([self._output], feed, self._run_opts)[0]
            if releaser is not None:
                releaser.ran()
        logits = np.asarray(logits, dtype=np.float64).reshape(-1)
        return (1.0 / (1.0 + np.exp(-logits))).tolist()

    def close(self) -> None:
        if self._releaser is not None:
            self._releaser.close()
        self._session = None


_reranker: CrossEncoder | None = None
_load_error: str | None = None
_lock = threading.Lock()


def load(fetch: bool = True) -> CrossEncoder | None:
    """Load the gate once (downloading the model first when `fetch`)."""
    global _reranker, _load_error
    if RERANK_MODE == "off":
        return None
    with _lock:
        if _reranker is not None:
            return _reranker
        try:
            if not model_present():
                if not fetch:
                    raise FileNotFoundError(f"no reranker model in {RERANK_ONNX_DIR}")
                download()
            _reranker = CrossEncoder()
            _load_error = None
        except Exception as e:
            _load_error = f"{type(e).__name__}: {str(e)[:300]}"
            logger.error("Reranker unavailable, search runs without the gate: %s", _load_error)
        return _reranker


def start() -> None:
    """Load the gate off the request path (server start-up)."""
    if RERANK_MODE == "off" or _reranker is not None:
        return
    threading.Thread(target=load, daemon=True, name="memory-index-rerank-init").start()


def get() -> CrossEncoder | None:
    """The loaded gate, or None (never loads: search must not block on it)."""
    return _reranker


def status() -> dict:
    """For /health: whether search is gated, on what, and why not."""
    r = _reranker
    return {
        "mode": RERANK_MODE,
        "loaded": r is not None,
        "device": r.device if r else None,
        "min": RERANK_MIN,
        "error": (r.device_error if r else None) or _load_error,
    }


def set_for_tests(reranker) -> None:
    """Install a stand-in (anything with score(query, texts)) or None."""
    global _reranker
    _reranker = reranker
