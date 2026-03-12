"""ChromaDB + CodeRankEmbed embedding setup.

GPU acceleration priority: CUDA (NVIDIA) > DirectML (AMD/Intel) > CPU.
Uses ONNX Runtime for inference; falls back to PyTorch if ONNX model not exported.
"""

import logging
import os
import pathlib
import sys
import gc

import chromadb
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings

from src.config import CHROMA_DIR, CODERANK_MODEL, CODERANK_QUERY_PREFIX, CODERANK_ONNX_DIR
from src.hardware import get_hardware_profile

logger = logging.getLogger(__name__)


def _register_nvidia_dll_dirs():
    """Register nvidia pip-wheel DLL directories so ONNX Runtime can find CUDA libraries.

    The nvidia-cublas-cu12, nvidia-cudnn-cu12, etc. pip packages install DLLs under
    site-packages/nvidia/<pkg>/bin/ but they're not on PATH by default.
    """
    if sys.platform != "win32":
        return
    nvidia_base = pathlib.Path(__file__).resolve().parents[2] / ".venv" / "Lib" / "site-packages" / "nvidia"
    if not nvidia_base.exists():
        return
    registered = 0
    for bin_dir in nvidia_base.glob("*/bin"):
        if bin_dir.is_dir() and any(bin_dir.glob("*.dll")):
            try:
                os.add_dll_directory(str(bin_dir))
            except OSError:
                pass
            # Also prepend to PATH — ORT's native transitive DLL loads use PATH, not AddDllDirectory
            os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
            registered += 1
    if registered:
        logger.debug("Registered %d nvidia DLL directories for CUDA", registered)


_register_nvidia_dll_dirs()

_client: chromadb.ClientAPI | None = None
_embedding_fns: dict[str, "CodeRankEmbedder"] = {}

# Detected backend after initialization (for status reporting)
_active_backends: dict[str, str] = {}

_ROLE_ENV_MAP = {
    "index": ("CODERANK_INDEX_BACKEND", "gpu"),
    "search": ("CODERANK_SEARCH_BACKEND", "cpu"),
}


def _normalize_role(role: str) -> str:
    role_key = (role or "index").lower()
    if role_key not in _ROLE_ENV_MAP:
        return "index"
    return role_key


def _session_init_strategy(role: str, mode: str) -> tuple[str, bool]:
    """Return session init strategy as (graph_opt_level, persist_optimized_model).

    graph_opt_level:
      - "all": ORT_ENABLE_ALL (best throughput)
      - "extended": ORT_ENABLE_EXTENDED (faster cold init, lower compile overhead)
      - "disabled": ORT_DISABLE_ALL (fastest cold init, for pre-optimized models)
    """
    role_key = _normalize_role(role)
    mode_key = (mode or "auto").lower()

    # Search runs frequently in fresh terminal processes. For CPU search sessions,
    # skip graph optimization entirely — load the pre-optimized model if available,
    # or accept slightly slower inference to avoid 30-40s cold-start stall.
    if role_key == "search" and mode_key == "cpu":
        return ("disabled", False)

    return ("all", True)


class _LightweightCpuProfile:
    """Minimal hardware profile for search CPU cold-start path.

    Avoids expensive GPU/PowerShell probing in fresh processes where search-only
    CPU inference does not need VRAM introspection.
    """

    def __init__(self):
        self.gpu_vram_mb = None
        self.gpu_free_vram_mb = None
        self.cpu_cores = os.cpu_count() or 4
        self.system_ram_mb = 8192
        self.vram_budget_mb = 4096.0
        self.is_gpu = False
        usable = max(2, int(self.cpu_cores * 0.50))
        self.intra_op_threads = usable
        self.inter_op_threads = max(1, usable // 4)
        self.embedding_batch_size = 100
        self.backend = "ONNX + CPU (forced)"

    def configure_for_backend(self, backend: str) -> None:
        self.backend = backend
        self.is_gpu = False

    def max_batch_for_seq(self, seq_len: int) -> int:
        if seq_len <= 256:
            return 16
        if seq_len <= 512:
            return 8
        return 4

    def seq_fits_in_vram(self, seq_len: int) -> bool:
        return True

    def summary(self) -> str:
        return (
            f"GPU: skipped (search CPU fast path) | CPU: {self.cpu_cores} cores | "
            f"RAM: ~{self.system_ram_mb} MB (default) | ONNX batch: heuristic "
            f"(256tok={self.max_batch_for_seq(256)}, 512tok={self.max_batch_for_seq(512)}, "
            f"1k={self.max_batch_for_seq(1024)}) | Threads: intra={self.intra_op_threads}, "
            f"inter={self.inter_op_threads} | Embedding batch: {self.embedding_batch_size}"
        )


def _use_lightweight_profile(role: str, mode: str) -> bool:
    return _normalize_role(role) == "search" and (mode or "auto").lower() == "cpu"


class _FastTokenizerWrapper:
    """Lightweight tokenizer using the `tokenizers` library directly.

    Avoids importing the heavy `transformers` package (~25s on Windows cold start,
    sometimes deadlocks in background threads). Provides the same __call__ API
    as HuggingFace AutoTokenizer for the subset used by _onnx_embed.
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


class CodeRankEmbedder(EmbeddingFunction[Documents]):
    """CodeRankEmbed embeddings (137M params, 768-dim).

    Uses ONNX Runtime for inference with GPU acceleration when available.
    Priority: CUDA (NVIDIA) > DirectML (AMD/Intel) > CPU.
    Falls back to PyTorch CPU if ONNX model not found.
    CLS-token pooling (first token) per model config.
    """

    def __init__(self, backend_mode: str = "auto", role: str = "index"):
        self._ort_session = None
        self._tokenizer = None
        self._pt_model = None  # fallback
        self.backend = "not initialized"
        self.backend_mode = backend_mode
        self.role = _normalize_role(role)
        if _use_lightweight_profile(self.role, self.backend_mode):
            self._hw = _LightweightCpuProfile()
            logger.info("Using lightweight hardware profile for search CPU cold-start path")
        else:
            self._hw = get_hardware_profile()

        onnx_path = CODERANK_ONNX_DIR / "model.onnx"
        if onnx_path.exists():
            self._init_onnx(str(onnx_path), backend_mode, self.role)
        else:
            self._init_pytorch()

        logger.info("Dynamic config: %s", self._hw.summary())

    def _init_onnx(self, onnx_path: str, backend_mode: str = "auto", role: str = "index"):
        """Load ONNX model with best available GPU acceleration."""
        logger.info("ONNX init start: role=%s mode=%s path=%s", role, backend_mode, onnx_path)
        logger.info("ONNX init: importing onnxruntime")
        import onnxruntime as ort
        logger.info("ONNX init: onnxruntime imported")

        # Guard: CPU-only onnxruntime and onnxruntime-directml share the same Python
        # namespace. If both are installed, CPU shadows DirectML and GPU silently breaks.
        # Skip this check for search path (CPU-only, doesn't care about GPU conflicts).
        if role != "search":
            try:
                from importlib.metadata import distributions
                ort_packages = [d.metadata["Name"] for d in distributions()
                               if d.metadata["Name"] and d.metadata["Name"].startswith("onnxruntime")]
                if "onnxruntime" in ort_packages and "onnxruntime-directml" in ort_packages:
                    logger.error(
                        "CONFLICTING PACKAGES: Both 'onnxruntime' (CPU) and 'onnxruntime-directml' "
                        "are installed. CPU version will shadow DirectML — GPU acceleration is BROKEN. "
                        "Fix: pip uninstall onnxruntime -y"
                    )
            except Exception:
                pass

        available = ort.get_available_providers()
        use_providers = []

        mode = (backend_mode or "auto").lower()
        if mode not in {"auto", "gpu", "cpu"}:
            logger.warning("Unknown backend_mode '%s', falling back to auto", backend_mode)
            mode = "auto"

        if mode == "cpu":
            use_providers = ["CPUExecutionProvider"]
            pending_backend = "ONNX + CPU (forced)"
        elif mode == "gpu":
            # Prefer GPU providers, fall back to CPU if unavailable.
            if "CUDAExecutionProvider" in available:
                use_providers.append("CUDAExecutionProvider")
            if "DmlExecutionProvider" in available:
                use_providers.append("DmlExecutionProvider")
            use_providers.append("CPUExecutionProvider")
            if "CUDAExecutionProvider" in use_providers:
                pending_backend = "ONNX + CUDA (NVIDIA GPU)"
            elif "DmlExecutionProvider" in use_providers:
                pending_backend = "ONNX + DirectML (AMD GPU)"
            else:
                pending_backend = "ONNX + CPU (GPU requested, unavailable)"
        else:
            # auto: Priority CUDA > DirectML > CPU
            if "CUDAExecutionProvider" in available:
                use_providers.append("CUDAExecutionProvider")
            if "DmlExecutionProvider" in available:
                use_providers.append("DmlExecutionProvider")
            use_providers.append("CPUExecutionProvider")
            if "CUDAExecutionProvider" in use_providers:
                pending_backend = "ONNX + CUDA (NVIDIA GPU)"
            elif "DmlExecutionProvider" in use_providers:
                pending_backend = "ONNX + DirectML (AMD GPU)"
            else:
                pending_backend = "ONNX + CPU"
        self._hw.configure_for_backend(pending_backend)

        # Select startup strategy by role.
        graph_opt_level, persist_optimized_model = _session_init_strategy(role, mode)

        # Search CPU fast path: load the pre-optimized CPU model (saved by index backend)
        # with all graph optimization disabled. Cuts cold-start from ~40s to ~3s.
        # Only load the CPU-specific model — CUDA-optimized models contain fused ops
        # (BiasSoftmax, FusedMatMul) that fail on CPUExecutionProvider.
        if graph_opt_level == "disabled":
            pre_optimized_cpu = CODERANK_ONNX_DIR / "model_optimized_cpu.onnx"
            if pre_optimized_cpu.exists():
                onnx_path = str(pre_optimized_cpu)
                logger.info("Search fast path: loading CPU-optimized model %s", pre_optimized_cpu)

        sess_opts = ort.SessionOptions()
        if graph_opt_level == "disabled":
            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            logger.info("ONNX session strategy: role=%s mode=%s, graph_opt=DISABLED", role, mode)
        elif graph_opt_level == "extended":
            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
            logger.info("ONNX session strategy: role=%s mode=%s, graph_opt=EXTENDED", role, mode)
        else:
            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            logger.info("ONNX session strategy: role=%s mode=%s, graph_opt=ALL", role, mode)

        # Thread counts: 0 = let ORT auto-tune (best for GPU), explicit for CPU-only
        if self._hw.intra_op_threads > 0:
            sess_opts.intra_op_num_threads = self._hw.intra_op_threads
            sess_opts.inter_op_num_threads = self._hw.inter_op_threads
            logger.info("ONNX threads: intra_op=%d, inter_op=%d (from %d cores)",
                         self._hw.intra_op_threads, self._hw.inter_op_threads, self._hw.cpu_cores)
        else:
            logger.info("ONNX threads: auto (GPU backend, %d cores available)", self._hw.cpu_cores)

        if persist_optimized_model:
            # Cache optimized graph to disk so subsequent loads can skip optimization.
            # Use backend-specific suffix — CUDA-optimized models contain fused ops
            # (BiasSoftmax, FusedMatMul) that fail on CPUExecutionProvider.
            backend_suffix = "cuda" if "cuda" in pending_backend.lower() else "cpu"
            optimized_path = CODERANK_ONNX_DIR / f"model_optimized_{backend_suffix}.onnx"
            sess_opts.optimized_model_filepath = str(optimized_path)
        else:
            logger.info("ONNX session strategy: role=%s mode=%s, skip optimized-model serialization", role, mode)

        self._ort_session = ort.InferenceSession(onnx_path, sess_opts, providers=use_providers)

        # Tokenizer: always use lightweight tokenizers lib (avoids ~25s transformers
        # import that can deadlock in background threads on Windows).
        tokenizer_json = CODERANK_ONNX_DIR / "tokenizer.json"
        if tokenizer_json.exists():
            logger.info("ONNX init: loading tokenizer via lightweight tokenizers library")
            self._tokenizer = _FastTokenizerWrapper(str(tokenizer_json))
        else:
            logger.info("ONNX init: tokenizer.json not found, falling back to transformers")
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(str(CODERANK_ONNX_DIR))

        # Determine which provider is actually active
        active = self._ort_session.get_providers()
        if "CUDAExecutionProvider" in active:
            self.backend = "ONNX + CUDA (NVIDIA GPU)"
        elif "DmlExecutionProvider" in active:
            self.backend = "ONNX + DirectML (AMD GPU)"
        else:
            self.backend = "ONNX + CPU"

        if mode == "cpu":
            self.backend = "ONNX + CPU (forced)"
        elif mode == "gpu" and self.backend == "ONNX + CPU":
            self.backend = "ONNX + CPU (GPU requested, unavailable)"

        logger.info("CodeRankEmbed loaded: %s", self.backend)

    def warmup(self):
        """Run a single dummy inference to initialize the GPU session.

        Keeps startup fast (~1 inference). Long-sequence DirectML failures are
        handled by the fallback chain in _onnx_embed (batch -> individual retry
        -> truncate to 512).
        """
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
                logger.warning("Warmup inference failed: %s", e)
            logger.info("CodeRankEmbed warmup complete")

    # Batch size buckets used for warmup shape enumeration.
    _WARMUP_BATCH_BUCKETS = (1, 2, 4, 8, 16, 32)

    def warmup_all_shapes(self, progress_callback=None):
        """Pre-compile GPU kernels for all (batch, seq_len) shape combinations.

        DirectML/CUDA compiles a kernel on first encounter of each unique
        (batch_size, seq_len) shape.  Without this, the first real inference
        at each shape stalls for 3-4 seconds.  By iterating all feasible
        bucket pairs at startup we pay that cost once, up front.

        Skipped for CPU backends (kernel compilation is instant).
        """
        if self._ort_session is None:
            return
        if "CPU" in self.backend and "CUDA" not in self.backend and "DirectML" not in self.backend:
            logger.info("Skipping full shape warmup (CPU backend — kernels compile instantly)")
            return

        import numpy as np
        import time as _time

        feasible = []
        skipped_seq = []
        for seq_len in self._SEQ_LEN_BUCKETS:
            # Skip seq lengths where even batch=1 exceeds the VRAM budget.
            # Attempting these causes the GPU allocator to grab all remaining
            # VRAM without releasing it, even when the inference fails.
            if not self._hw.seq_fits_in_vram(seq_len):
                skipped_seq.append(seq_len)
                continue
            max_batch = self._hw.max_batch_for_seq(seq_len)
            for batch_size in self._WARMUP_BATCH_BUCKETS:
                if batch_size <= max_batch:
                    feasible.append((batch_size, seq_len))

        if skipped_seq:
            logger.info("Skipping warmup for seq lengths %s (exceed VRAM budget)", skipped_seq)
        logger.info("Warming up %d GPU kernel shapes (%d seq × %d batch buckets, filtered by VRAM)...",
                     len(feasible), len(self._SEQ_LEN_BUCKETS) - len(skipped_seq), len(self._WARMUP_BATCH_BUCKETS))

        t_total_start = _time.perf_counter()
        succeeded = 0
        failed = 0
        for i, (batch_size, seq_len) in enumerate(feasible):
            if progress_callback:
                progress_callback(i, len(feasible),
                                  f"Warmup shape {i+1}/{len(feasible)}: batch={batch_size}, seq={seq_len}")
            dummy_ids = np.zeros((batch_size, seq_len), dtype=np.int64)
            dummy_mask = np.zeros((batch_size, seq_len), dtype=np.int64)
            feed = {"input_ids": dummy_ids, "attention_mask": dummy_mask}
            t_shape = _time.perf_counter()
            try:
                self._ort_session.run(None, feed)
                dt = _time.perf_counter() - t_shape
                succeeded += 1
                if dt > 0.5:
                    logger.info("  warmup shape batch=%d seq=%d: %.2fs (slow)", batch_size, seq_len, dt)
            except Exception:
                dt = _time.perf_counter() - t_shape
                failed += 1
                if dt > 0.5:
                    logger.info("  warmup shape batch=%d seq=%d: %.2fs (failed, slow)", batch_size, seq_len, dt)

        t_total = _time.perf_counter() - t_total_start
        logger.info("GPU kernel warmup complete: %d/%d succeeded, %d failed, %.1fs total",
                     succeeded, len(feasible), failed, t_total)

    def _init_pytorch(self):
        """Fallback: load via sentence-transformers (PyTorch CPU)."""
        from sentence_transformers import SentenceTransformer
        logger.warning(
            "CodeRankEmbed: ONNX model not found at %s — using PyTorch CPU (slower). "
            "Run setup.bat to export the ONNX model for GPU acceleration.",
            CODERANK_ONNX_DIR,
        )
        self._pt_model = SentenceTransformer(CODERANK_MODEL, trust_remote_code=True)
        self.backend = "PyTorch CPU (no ONNX export — run setup.bat)"
        self._hw.configure_for_backend(self.backend)
        logger.info("CodeRankEmbed loaded: %s", self.backend)

    # Approximate characters-per-token ratio for code (BPE tokenizers).
    # Used to estimate token count from char length for adaptive batch sizing.
    _CHARS_PER_TOKEN = 3.5

    def _estimate_tokens(self, char_len: int) -> int:
        """Estimate token count from character length."""
        return min(int(char_len / self._CHARS_PER_TOKEN) + 16, 8192)

    def _max_safe_seq_len(self) -> int:
        """Return the largest sequence length bucket that fits in the VRAM budget.

        Used as max_length for the tokenizer to proactively truncate text that
        would blow past VRAM, instead of attempting inference and relying on
        the GPU allocator to fail gracefully (it doesn't — CUDA's caching
        allocator grabs all remaining VRAM and never releases it).
        """
        for bucket in reversed(self._SEQ_LEN_BUCKETS):
            if self._hw.seq_fits_in_vram(bucket):
                return bucket
        return self._SEQ_LEN_BUCKETS[0]  # floor at smallest bucket (64)

    # Fixed sequence length buckets to limit unique GPU kernel compilations.
    # DirectML compiles and caches a kernel per unique (batch, seq_len) shape.
    # Without bucketing, hundreds of unique shapes leak VRAM on large repos.
    _SEQ_LEN_BUCKETS = (64, 128, 256, 512, 1024, 2048, 4096, 8192)

    @staticmethod
    def _bucket_pad(input_ids, attention_mask, buckets=_SEQ_LEN_BUCKETS):
        """Pad sequence length to the next bucket boundary.

        Ensures the GPU only sees a small set of fixed sequence lengths,
        preventing unbounded DirectML kernel cache growth.
        """
        import numpy as np
        seq_len = input_ids.shape[1]
        target = seq_len
        for b in buckets:
            if b >= seq_len:
                target = b
                break
        if target == seq_len:
            return input_ids, attention_mask
        pad_width = target - seq_len
        input_ids = np.pad(input_ids, ((0, 0), (0, pad_width)), constant_values=0)
        attention_mask = np.pad(attention_mask, ((0, 0), (0, pad_width)), constant_values=0)
        return input_ids, attention_mask

    def _onnx_embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts via ONNX Runtime. Adaptive batched inference with CLS-token pooling.

        Sorts by length to minimize padding waste, then dynamically sizes each batch
        based on the longest text's estimated token count and available VRAM.
        Short texts get large batches, long texts get small batches.
        Results are returned in original order. If a GPU batch fails (e.g. DirectML
        Reshape error at unusual sequence lengths), falls back to embedding each text
        individually at batch_size=1.

        Sequence lengths are bucketed to fixed sizes (64, 128, ..., 8192) to bound
        the number of unique GPU kernel compilations on DirectML.

        Tokenizer max_length is capped to the largest sequence length that fits
        in the VRAM budget, preventing oversized sequences from causing the GPU
        allocator to grab all remaining VRAM.
        """
        import numpy as np

        if not texts:
            return []

        # Cap tokenizer length to what actually fits in VRAM.  On a 4 GB GPU
        # this may be 1024 instead of 8192, preventing OOM-induced VRAM leaks.
        max_tok_len = self._max_safe_seq_len()

        # Sort by length to group similar-length texts (less padding waste)
        indexed = sorted(enumerate(texts), key=lambda x: len(x[1]))
        results = [None] * len(texts)

        i = 0
        while i < len(indexed):
            # Determine adaptive batch size from the longest text in the candidate window
            remaining = len(indexed) - i
            # Estimate tokens from the longest text at the far end of a max-sized window
            max_window = min(32, remaining)
            longest_chars = len(indexed[i + max_window - 1][1])
            est_tokens = min(self._estimate_tokens(longest_chars), max_tok_len)
            safe_batch = self._hw.max_batch_for_seq(est_tokens)

            # If safe_batch < max_window, shrink window and re-check
            # (shorter window means shorter longest text, possibly allowing more)
            if safe_batch < max_window:
                actual_size = min(safe_batch, remaining)
                longest_chars = len(indexed[i + actual_size - 1][1])
                est_tokens = min(self._estimate_tokens(longest_chars), max_tok_len)
                actual_size = min(self._hw.max_batch_for_seq(est_tokens), remaining)
            else:
                actual_size = max_window

            actual_size = max(1, actual_size)
            batch = indexed[i:i + actual_size]
            i += actual_size
            batch_indices, batch_texts = zip(*batch)

            inp = self._tokenizer(list(batch_texts), return_tensors="np",
                                  padding=True, truncation=True, max_length=max_tok_len)
            ids = inp["input_ids"].astype(np.int64)
            mask = inp["attention_mask"].astype(np.int64)
            ids, mask = self._bucket_pad(ids, mask)
            feed = {"input_ids": ids, "attention_mask": mask}
            try:
                out = self._ort_session.run(None, feed)[0]  # (batch, seq, 768)
            except Exception as e:
                # DirectML can fail on certain sequence length × batch shape combinations.
                # Fall back to embedding each text individually (batch_size=1).
                seq_len = ids.shape[1]
                logger.warning("GPU batch failed (seq_len=%d, batch=%d), falling back to individual: %s",
                               seq_len, len(batch_texts), str(e)[:120])
                for j, (orig_idx, text) in enumerate(zip(batch_indices, batch_texts)):
                    inp1 = self._tokenizer([text], return_tensors="np",
                                           padding=True, truncation=True, max_length=max_tok_len)
                    ids1 = inp1["input_ids"].astype(np.int64)
                    mask1 = inp1["attention_mask"].astype(np.int64)
                    ids1, mask1 = self._bucket_pad(ids1, mask1)
                    feed1 = {"input_ids": ids1, "attention_mask": mask1}
                    embedded = False
                    # DirectML often succeeds on retry — failed inferences still prime
                    # the kernel cache, so the next attempt at the same shape works.
                    for attempt in range(3):
                        try:
                            out1 = self._ort_session.run(None, feed1)[0]
                            results[orig_idx] = out1[0, 0, :].tolist()
                            embedded = True
                            break
                        except Exception:
                            if attempt < 2:
                                continue
                    if not embedded:
                        # Last resort: truncate to 512 tokens
                        sl = ids1.shape[1]
                        fallback_len = min(512, max_tok_len)
                        logger.warning("Embed failed after 3 retries (seq_len=%d), truncating to %d",
                                       sl, fallback_len)
                        inp_short = self._tokenizer([text], return_tensors="np",
                                                    padding=True, truncation=True, max_length=fallback_len)
                        ids_s = inp_short["input_ids"].astype(np.int64)
                        mask_s = inp_short["attention_mask"].astype(np.int64)
                        ids_s, mask_s = self._bucket_pad(ids_s, mask_s)
                        feed_short = {"input_ids": ids_s, "attention_mask": mask_s}
                        out_short = self._ort_session.run(None, feed_short)[0]
                        results[orig_idx] = out_short[0, 0, :].tolist()
                continue

            for j, orig_idx in enumerate(batch_indices):
                results[orig_idx] = out[j, 0, :].tolist()  # CLS token

        return results

    def __call__(self, input: Documents) -> Embeddings:
        """Embed documents (no prefix). Called by ChromaDB at add time."""
        if self._ort_session is not None:
            return self._onnx_embed(list(input))
        embeddings = self._pt_model.encode(list(input), show_progress_bar=False)
        return embeddings.tolist()

    def embed_queries(self, queries: list[str]) -> list[list[float]]:
        """Embed queries WITH the code search prefix. Used at query time."""
        prefixed = [CODERANK_QUERY_PREFIX + q for q in queries]
        if self._ort_session is not None:
            return self._onnx_embed(prefixed)
        embeddings = self._pt_model.encode(prefixed, show_progress_bar=False)
        return embeddings.tolist()

    def close(self) -> None:
        """Release model/session references so memory can be reclaimed."""
        self._ort_session = None
        self._tokenizer = None
        self._pt_model = None
        self.backend = "released"


def _resolve_role_mode(role: str) -> str:
    role_key = _normalize_role(role)
    env_name, default_mode = _ROLE_ENV_MAP[role_key]
    mode = os.environ.get(env_name, default_mode).strip().lower()
    if mode not in {"auto", "gpu", "cpu"}:
        logger.warning("Invalid %s=%r, using default '%s'", env_name, mode, default_mode)
        mode = default_mode
    return mode


def get_embedding_function(role: str = "index", mode: str | None = None) -> CodeRankEmbedder:
    """Get or create a role/mode-specific singleton CodeRankEmbedder."""
    role_key = _normalize_role(role)
    resolved_mode = (mode or _resolve_role_mode(role_key)).lower()
    if resolved_mode not in {"auto", "gpu", "cpu"}:
        resolved_mode = "auto"
    key = f"{role_key}:{resolved_mode}"
    if key not in _embedding_fns:
        _embedding_fns[key] = CodeRankEmbedder(backend_mode=resolved_mode, role=role_key)
        _active_backends[key] = _embedding_fns[key].backend
    return _embedding_fns[key]


def release_embedding_function(role: str = "index", mode: str | None = None) -> None:
    """Release a role/mode embedding singleton (best-effort memory reclamation)."""
    role_key = _normalize_role(role)
    resolved_mode = (mode or _resolve_role_mode(role_key)).lower()
    key = f"{role_key}:{resolved_mode}"
    ef = _embedding_fns.pop(key, None)
    if ef is not None:
        try:
            ef.close()
        finally:
            gc.collect()
    _active_backends.pop(key, None)


def get_active_backend(role: str = "index", mode: str | None = None) -> str:
    """Return active backend for a role/mode if initialized, else 'not initialized'."""
    role_key = _normalize_role(role)
    resolved_mode = (mode or _resolve_role_mode(role_key)).lower()
    key = f"{role_key}:{resolved_mode}"
    return _active_backends.get(key, "not initialized")


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
