"""ChromaDB + CodeRankEmbed embedding setup (CPU-only).

Memory operations embed one observation at a time (~10ms), so GPU acceleration
adds complexity with no practical benefit. This is a simplified CPU-only
embedder derived from code-index's full GPU/CPU version.
"""

import logging
import os
import pathlib
import sys
import gc

import chromadb
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings

from src.config import CHROMA_DIR, CODERANK_MODEL, CODERANK_QUERY_PREFIX, CODERANK_ONNX_DIR

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
                 max_length: int = 8192) -> dict:
        import numpy as np

        if truncation:
            self._tok.enable_truncation(max_length=max_length)
        else:
            self._tok.no_truncation()

        if padding:
            self._tok.enable_padding(pad_id=0)
        else:
            self._tok.no_padding()

        encoded = self._tok.encode_batch(texts)

        ids = np.array([e.ids for e in encoded], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)

        return {"input_ids": ids, "attention_mask": mask}


_client: chromadb.ClientAPI | None = None
_embedding_fn: "CodeRankEmbedder | None" = None
_active_backend: str = "not initialized"


class CodeRankEmbedder(EmbeddingFunction[Documents]):
    """CodeRankEmbed embeddings (137M params, 768-dim), CPU-only ONNX.

    CLS-token pooling (first token) per model config.
    Falls back to PyTorch CPU if ONNX model not found.
    """

    def __init__(self):
        self._ort_session = None
        self._tokenizer = None
        self._pt_model = None
        self.backend = "not initialized"

        onnx_path = CODERANK_ONNX_DIR / "model.onnx"
        if onnx_path.exists():
            self._init_onnx(str(onnx_path))
        else:
            self._init_pytorch()

    def _init_onnx(self, onnx_path: str):
        """Load ONNX model with CPU execution provider."""
        logger.info("ONNX init: importing onnxruntime (CPU-only)")
        import onnxruntime as ort

        # Use pre-optimized CPU model if available (saves ~30s cold-start)
        pre_optimized = CODERANK_ONNX_DIR / "model_optimized_cpu.onnx"
        if pre_optimized.exists():
            onnx_path = str(pre_optimized)
            logger.info("Using pre-optimized CPU model: %s", pre_optimized)

        sess_opts = ort.SessionOptions()
        if str(onnx_path).endswith("_cpu.onnx"):
            # Pre-optimized: skip graph optimization
            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        else:
            # First load: optimize and persist for next time
            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            optimized_path = CODERANK_ONNX_DIR / "model_optimized_cpu.onnx"
            sess_opts.optimized_model_filepath = str(optimized_path)

        # CPU thread tuning
        cores = os.cpu_count() or 4
        usable = max(2, int(cores * 0.5))
        sess_opts.intra_op_num_threads = usable
        sess_opts.inter_op_num_threads = max(1, usable // 4)

        self._ort_session = ort.InferenceSession(
            onnx_path, sess_opts, providers=["CPUExecutionProvider"]
        )

        # Tokenizer
        tokenizer_json = CODERANK_ONNX_DIR / "tokenizer.json"
        if tokenizer_json.exists():
            self._tokenizer = _FastTokenizerWrapper(str(tokenizer_json))
        else:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(str(CODERANK_ONNX_DIR))

        self.backend = "ONNX + CPU"
        logger.info("CodeRankEmbed loaded: %s (%d threads)", self.backend, usable)

    def _init_pytorch(self):
        """Fallback: load via sentence-transformers (PyTorch CPU)."""
        from sentence_transformers import SentenceTransformer
        logger.warning(
            "ONNX model not found at %s — using PyTorch CPU (slower). "
            "Run setup.bat to export the ONNX model.",
            CODERANK_ONNX_DIR,
        )
        self._pt_model = SentenceTransformer(CODERANK_MODEL, trust_remote_code=True)
        self.backend = "PyTorch CPU (no ONNX export)"
        logger.info("CodeRankEmbed loaded: %s", self.backend)

    def warmup(self):
        """Run a single dummy inference to initialize the session."""
        if self._ort_session is not None:
            import numpy as np
            dummy = self._tokenizer(["warmup"], return_tensors="np",
                                    padding=True, truncation=True, max_length=8192)
            feed = {
                "input_ids": dummy["input_ids"].astype(np.int64),
                "attention_mask": dummy["attention_mask"].astype(np.int64),
            }
            try:
                self._ort_session.run(None, feed)
            except Exception as e:
                logger.warning("Warmup failed: %s", e)
            logger.info("CodeRankEmbed warmup complete")

    def _onnx_embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts via ONNX Runtime CPU. Simple batched inference with CLS pooling."""
        import numpy as np

        if not texts:
            return []

        # For memory-index, texts are typically 1-5 items. No adaptive batching needed.
        inp = self._tokenizer(list(texts), return_tensors="np",
                              padding=True, truncation=True, max_length=8192)
        ids = inp["input_ids"].astype(np.int64)
        mask = inp["attention_mask"].astype(np.int64)
        feed = {"input_ids": ids, "attention_mask": mask}

        try:
            out = self._ort_session.run(None, feed)[0]  # (batch, seq, 768)
        except Exception as e:
            # Fallback: embed one at a time
            logger.warning("Batch embed failed, falling back to individual: %s", str(e)[:120])
            results = []
            for text in texts:
                inp1 = self._tokenizer([text], return_tensors="np",
                                       padding=True, truncation=True, max_length=512)
                ids1 = inp1["input_ids"].astype(np.int64)
                mask1 = inp1["attention_mask"].astype(np.int64)
                out1 = self._ort_session.run(None, {"input_ids": ids1, "attention_mask": mask1})[0]
                results.append(out1[0, 0, :].tolist())
            return results

        return [out[i, 0, :].tolist() for i in range(out.shape[0])]  # CLS token

    def __call__(self, input: Documents) -> Embeddings:
        """Embed documents (no prefix). Called by ChromaDB at add time."""
        if self._ort_session is not None:
            return self._onnx_embed(list(input))
        embeddings = self._pt_model.encode(list(input), show_progress_bar=False)
        return embeddings.tolist()

    def embed_queries(self, queries: list[str]) -> list[list[float]]:
        """Embed queries WITH the search prefix. Used at query time."""
        prefixed = [CODERANK_QUERY_PREFIX + q for q in queries]
        if self._ort_session is not None:
            return self._onnx_embed(prefixed)
        embeddings = self._pt_model.encode(prefixed, show_progress_bar=False)
        return embeddings.tolist()

    def close(self) -> None:
        """Release model/session references."""
        self._ort_session = None
        self._tokenizer = None
        self._pt_model = None
        self.backend = "released"


def get_embedding_function(role: str = "index", mode: str | None = None) -> CodeRankEmbedder:
    """Get or create the singleton CodeRankEmbedder (CPU-only, one instance)."""
    global _embedding_fn, _active_backend
    if _embedding_fn is None:
        _embedding_fn = CodeRankEmbedder()
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
