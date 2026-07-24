"""ChromaDB + EmbeddingGemma-300m embedding setup (CPU-only).

Memory operations embed one observation at a time, so GPU acceleration
adds complexity with no practical benefit. Uses the onnx-community q8 export,
whose graph bakes in mean pooling, the dense projections, and L2 normalization —
the session outputs a finished `sentence_embedding` tensor directly.
"""

import logging
import os
import pathlib
import sys
import gc

import chromadb
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings

from src.config import (
    CHROMA_DIR,
    EMBED_ONNX_DIR,
    EMBED_ONNX_FILENAME,
    EMBED_PT_MODEL,
    EMBED_QUERY_PREFIX,
    EMBED_DOC_PREFIX,
    EMBED_MAX_TOKENS,
)

logger = logging.getLogger(__name__)


class _FastTokenizerWrapper:
    """Lightweight tokenizer using the `tokenizers` library directly.

    Avoids importing the heavy `transformers` package (~25s on Windows cold start,
    sometimes deadlocks in background threads).
    """

    def __init__(self, tokenizer_json_path: str):
        from tokenizers import Tokenizer
        self._tok = Tokenizer.from_file(tokenizer_json_path)

    def __call__(self, texts: list[str], return_tensors: str = "np",
                 padding: bool = True, truncation: bool = True,
                 max_length: int = EMBED_MAX_TOKENS) -> dict:
        import numpy as np

        if truncation:
            self._tok.enable_truncation(max_length=max_length)
        else:
            self._tok.no_truncation()

        if padding:
            # Gemma pad token id is 0 (<pad>)
            self._tok.enable_padding(pad_id=0)
        else:
            self._tok.no_padding()

        encoded = self._tok.encode_batch(texts)

        ids = np.array([e.ids for e in encoded], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)

        return {"input_ids": ids, "attention_mask": mask}


_client: chromadb.ClientAPI | None = None
_embedding_fn: "GemmaEmbedder | None" = None
_active_backend: str = "not initialized"


class GemmaEmbedder(EmbeddingFunction[Documents]):
    """EmbeddingGemma-300m embeddings (308M params, 768-dim), CPU-only ONNX q8.

    The ONNX graph outputs L2-normalized sentence embeddings directly.
    Falls back to PyTorch CPU if the ONNX model is not found.
    """

    def __init__(self):
        self._ort_session = None
        self._tokenizer = None
        self._pt_model = None
        self.backend = "not initialized"

        onnx_path = EMBED_ONNX_DIR / EMBED_ONNX_FILENAME
        if onnx_path.exists():
            self._init_onnx(str(onnx_path))
        else:
            self._init_pytorch()

    def _init_onnx(self, onnx_path: str):
        """Load ONNX model with CPU execution provider."""
        logger.info("ONNX init: importing onnxruntime (CPU-only)")
        import onnxruntime as ort

        sess_opts = ort.SessionOptions()
        # No persist-optimized-graph flow here: the q8 model uses external
        # weight data, and session creation is only a few seconds.
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # CPU thread tuning
        cores = os.cpu_count() or 4
        usable = max(2, int(cores * 0.5))
        sess_opts.intra_op_num_threads = usable
        sess_opts.inter_op_num_threads = max(1, usable // 4)

        self._ort_session = ort.InferenceSession(
            onnx_path, sess_opts, providers=["CPUExecutionProvider"]
        )

        # Tokenizer
        tokenizer_json = EMBED_ONNX_DIR / "tokenizer.json"
        if tokenizer_json.exists():
            self._tokenizer = _FastTokenizerWrapper(str(tokenizer_json))
        else:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(str(EMBED_ONNX_DIR))

        self.backend = "ONNX + CPU"
        logger.info("EmbeddingGemma loaded: %s (%d threads)", self.backend, usable)

    def _init_pytorch(self):
        """Fallback: load via sentence-transformers (PyTorch CPU).

        Prefixes are applied manually before encode(), so encode() must not
        also apply the model's built-in prompts.
        """
        from sentence_transformers import SentenceTransformer
        logger.warning(
            "ONNX model not found at %s — using PyTorch CPU (slower, and %s is "
            "a gated repo requiring HF login). Run scripts/download_model.py "
            "to fetch the ONNX model.",
            EMBED_ONNX_DIR, EMBED_PT_MODEL,
        )
        self._pt_model = SentenceTransformer(EMBED_PT_MODEL)
        self.backend = "PyTorch CPU (no ONNX download)"
        logger.info("EmbeddingGemma loaded: %s", self.backend)

    def warmup(self):
        """Run a single dummy inference to initialize the session."""
        if self._ort_session is not None:
            import numpy as np
            dummy = self._tokenizer(["warmup"], return_tensors="np",
                                    padding=True, truncation=True,
                                    max_length=EMBED_MAX_TOKENS)
            feed = {
                "input_ids": dummy["input_ids"].astype(np.int64),
                "attention_mask": dummy["attention_mask"].astype(np.int64),
            }
            try:
                self._ort_session.run(["sentence_embedding"], feed)
            except Exception as e:
                logger.warning("Warmup failed: %s", e)
            logger.info("EmbeddingGemma warmup complete")

    def _onnx_embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts via ONNX Runtime CPU. The graph pools and normalizes."""
        import numpy as np

        if not texts:
            return []

        # For memory-index, texts are typically 1-5 items. No adaptive batching needed.
        inp = self._tokenizer(list(texts), return_tensors="np",
                              padding=True, truncation=True,
                              max_length=EMBED_MAX_TOKENS)
        ids = inp["input_ids"].astype(np.int64)
        mask = inp["attention_mask"].astype(np.int64)
        feed = {"input_ids": ids, "attention_mask": mask}

        try:
            out = self._ort_session.run(["sentence_embedding"], feed)[0]  # (batch, 768)
        except Exception as e:
            # Fallback: embed one at a time
            logger.warning("Batch embed failed, falling back to individual: %s", str(e)[:120])
            results = []
            for text in texts:
                inp1 = self._tokenizer([text], return_tensors="np",
                                       padding=True, truncation=True,
                                       max_length=EMBED_MAX_TOKENS)
                ids1 = inp1["input_ids"].astype(np.int64)
                mask1 = inp1["attention_mask"].astype(np.int64)
                out1 = self._ort_session.run(
                    ["sentence_embedding"],
                    {"input_ids": ids1, "attention_mask": mask1})[0]
                results.append(out1[0].tolist())
            return results

        return [out[i].tolist() for i in range(out.shape[0])]

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if self._ort_session is not None:
            return self._onnx_embed(texts)
        embeddings = self._pt_model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()

    def __call__(self, input: Documents) -> Embeddings:
        """Embed documents with the document prefix. Called by ChromaDB at add time."""
        return self._embed([EMBED_DOC_PREFIX + t for t in input])

    def embed_queries(self, queries: list[str]) -> list[list[float]]:
        """Embed queries with the retrieval query prefix. Used at query time."""
        return self._embed([EMBED_QUERY_PREFIX + q for q in queries])

    def close(self) -> None:
        """Release model/session references."""
        self._ort_session = None
        self._tokenizer = None
        self._pt_model = None
        self.backend = "released"


def get_embedding_function(role: str = "index", mode: str | None = None) -> GemmaEmbedder:
    """Get or create the singleton GemmaEmbedder (CPU-only, one instance)."""
    global _embedding_fn, _active_backend
    if _embedding_fn is None:
        _embedding_fn = GemmaEmbedder()
        _active_backend = _embedding_fn.backend
    return _embedding_fn


def release_embedding_function(role: str = "index", mode: str | None = None) -> None:
    """Release the embedding singleton."""
    global _embedding_fn, _active_backend
    if _embedding_fn is not None:
        try:
            _embedding_fn.close()
        finally:
            _embedding_fn = None
            _active_backend = "not initialized"
            gc.collect()


def get_active_backend(role: str = "index", mode: str | None = None) -> str:
    """Return active backend if initialized, else 'not initialized'."""
    return _active_backend


def get_chroma_client() -> chromadb.ClientAPI:
    """Get or create the singleton ChromaDB PersistentClient."""
    global _client
    if _client is None:
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return _client


def get_collection(collection_name: str) -> chromadb.Collection:
    """Get or create a ChromaDB collection without attaching an embedder."""
    client = get_chroma_client()
    return client.get_or_create_collection(name=collection_name)
