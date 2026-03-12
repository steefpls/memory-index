"""Dynamic hardware detection for optimal resource utilization.

Detects GPU VRAM, CPU core count, and system RAM at startup.
Computes optimal batch sizes and thread counts targeting ~50% utilization.
"""

import logging
import os
import subprocess
import sys

logger = logging.getLogger(__name__)

# Target utilization fraction — don't assume the full GPU is available.
# Other processes (desktop compositor, display driver, other apps) use VRAM.
TARGET_UTILIZATION = 0.50

# DirectML (AMD/Intel) uses WDDM memory pools that retain allocations for
# every unique (batch, seq_len) shape and NEVER release them until the ONNX
# session is destroyed.  The DIRECTML_MAX_BATCH cap (below) is the primary
# defense against VRAM explosion; utilization can match CUDA since the batch
# cap keeps per-shape pools small regardless of budget.
DIRECTML_UTILIZATION = 0.50

# DirectML max batch cap.  Even with lower utilization, short sequences
# (64-256 tokens) compute huge raw batch sizes (100+) that get capped at 32.
# Each shape's retained pool scales with batch size, so capping at 8 across
# the board keeps total retained VRAM to ~8 GB instead of ~20 GB.
DIRECTML_MAX_BATCH = 8

# CodeRankEmbed float16 model size in MB (approximate)
_MODEL_SIZE_MB = 274

# CodeRankEmbed architecture constants (BERT-like, ~137M params)
_NUM_LAYERS = 12
_NUM_HEADS = 12
_HIDDEN_DIM = 768

# Per-layer VRAM cost formula (bytes):
#   attention: batch × heads × seq² × 2 (fp16)  = batch × seq² × 24
#   activations/FFN: batch × seq × hidden × ~8   = batch × seq × 6144
#   total per layer: batch × (seq² × 24 + seq × 6144)
#   all layers: batch × (seq² × 288 + seq × 73728)
#
# Safety factor of 2x accounts for ONNX Runtime overhead, optimizer state,
# workspace buffers, and fragmentation.
_VRAM_SAFETY_FACTOR = 2.0
_ATTN_BYTES_PER_LAYER = _NUM_HEADS * 2  # 24 bytes per (seq² element) per layer
_FFN_BYTES_PER_LAYER = _HIDDEN_DIM * 8  # 6144 bytes per (seq element) per layer
_ATTN_COEFF = _ATTN_BYTES_PER_LAYER * _NUM_LAYERS  # 288
_FFN_COEFF = _FFN_BYTES_PER_LAYER * _NUM_LAYERS     # 73728


def detect_gpu_vram_mb() -> int | None:
    """Detect total GPU VRAM in MB.

    Uses nvidia-smi for NVIDIA, falls back to Windows registry (64-bit accurate)
    then WMI (UINT32, capped ~4 GB) for AMD/Intel.
    Returns None if detection fails.
    """
    # Try nvidia-smi first (most accurate for NVIDIA)
    vram = _detect_vram_nvidia_smi()
    if vram is not None:
        return vram

    if sys.platform == "win32":
        # Registry provides 64-bit VRAM (accurate for >4 GB GPUs like RX 7900 XTX).
        vram = _detect_vram_registry()
        if vram is not None:
            return vram

        # WMI fallback (AdapterRAM is UINT32, capped at ~4 GB).
        vram = _detect_vram_wmi()
        if vram is not None:
            return vram

    return None


def _detect_vram_nvidia_smi() -> int | None:
    """Query nvidia-smi for total GPU memory in MB."""
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL, timeout=10,
        )
        # May return multiple GPUs — take the first
        for line in output.strip().splitlines():
            val = line.strip()
            if val.isdigit():
                return int(val)
    except (FileNotFoundError, subprocess.SubprocessError):
        pass
    return None


def _detect_vram_registry() -> int | None:
    """Read 64-bit VRAM from Windows display adapter registry keys.

    The HardwareInformation.qwMemorySize registry value is a QWORD (64-bit)
    that accurately reports VRAM for GPUs with >4 GB, unlike WMI's UINT32
    AdapterRAM which overflows.  Takes the largest value across all adapters
    (discrete GPU will have more VRAM than integrated).

    Uses subprocess.run (not check_output) because some adapter entries may
    lack this key, causing PowerShell to exit non-zero even with
    -ErrorAction SilentlyContinue.  We still get valid output on stdout.
    """
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             'Get-ItemProperty -Path '
             '"HKLM:\\SYSTEM\\ControlSet001\\Control\\Class'
             '\\{4d36e968-e325-11ce-bfc1-08002be10318}\\0*" '
             '-Name "HardwareInformation.qwMemorySize" '
             '-ErrorAction SilentlyContinue '
             '| ForEach-Object { $_."HardwareInformation.qwMemorySize" }'],
            text=True, capture_output=True, timeout=10,
        )
        best = 0
        for line in result.stdout.strip().splitlines():
            val = line.strip()
            if val.isdigit():
                best = max(best, int(val))
        if best > 0:
            return best // (1024 * 1024)
    except (FileNotFoundError, subprocess.SubprocessError):
        pass
    return None


def _detect_vram_wmi() -> int | None:
    """Query Windows WMI for GPU adapter RAM in MB.

    Filters out virtual display adapters (Parsec, Remote Desktop, etc.) that
    report zero/null AdapterRAM.  Note: AdapterRAM is UINT32, so GPUs with
    >4 GB VRAM will show a truncated value (~4095 MB).
    """
    try:
        output = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_VideoController"
             " | Where-Object { $_.AdapterRAM -gt 0 }"
             " | Sort-Object AdapterRAM -Descending"
             " | Select-Object -First 1).AdapterRAM"],
            text=True, stderr=subprocess.DEVNULL, timeout=10,
        )
        val = output.strip()
        if val.isdigit():
            bytes_val = int(val)
            if bytes_val > 0:
                return bytes_val // (1024 * 1024)
    except (FileNotFoundError, subprocess.SubprocessError):
        pass
    return None


def detect_gpu_free_vram_mb() -> int | None:
    """Detect free/available GPU VRAM in MB (NVIDIA only).

    Returns None for non-NVIDIA GPUs or if detection fails.
    """
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL, timeout=10,
        )
        for line in output.strip().splitlines():
            val = line.strip()
            if val.isdigit():
                return int(val)
    except (FileNotFoundError, subprocess.SubprocessError):
        pass
    return None


def detect_cpu_cores() -> int:
    """Detect number of logical CPU cores."""
    return os.cpu_count() or 4


def detect_system_ram_mb() -> int:
    """Detect total system RAM in MB."""
    if sys.platform == "win32":
        try:
            output = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"],
                text=True, stderr=subprocess.DEVNULL, timeout=10,
            )
            val = output.strip()
            if val.isdigit():
                return int(val) // (1024 * 1024)
        except (FileNotFoundError, subprocess.SubprocessError):
            pass
    else:
        try:
            import resource
            # Not available on all platforms, but try
            mem_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
            return mem_bytes // (1024 * 1024)
        except (AttributeError, ValueError):
            pass
    return 8192  # conservative 8 GB default


def compute_vram_budget_mb(
    vram_total_mb: int | None,
    vram_free_mb: int | None,
    backend: str,
) -> float:
    """Compute usable VRAM budget for inference batches (excludes model weights).

    Uses FREE VRAM when available (accounts for other processes), falling
    back to a conservative fraction of total VRAM.

    Args:
        vram_total_mb: Total GPU VRAM in MB, or None if unknown.
        vram_free_mb: Free GPU VRAM in MB, or None if unknown.
        backend: Active backend string (used to detect GPU vs CPU).

    Returns:
        Available VRAM in MB for batch inference. Returns system RAM budget for CPU.
    """
    is_gpu = "CUDA" in backend or "DirectML" in backend

    if is_gpu:
        is_directml = "DirectML" in backend
        utilization = DIRECTML_UTILIZATION if is_directml else TARGET_UTILIZATION
        if vram_free_mb is not None:
            # Free VRAM already excludes loaded model weights, so don't subtract
            # _MODEL_SIZE_MB again. The 2x safety factor covers the ~274 MB uncertainty
            # during the initial call (before model load).
            available = vram_free_mb * utilization
        elif vram_total_mb is not None:
            available = (vram_total_mb * utilization) - _MODEL_SIZE_MB
        else:
            available = 1500.0  # conservative ~1.5 GB default
        return max(100.0, available)
    else:
        ram_mb = detect_system_ram_mb()
        available = (ram_mb * TARGET_UTILIZATION) - _MODEL_SIZE_MB
        return max(100.0, available)


def _raw_batch_for_seq_len(seq_len: int, vram_budget_mb: float) -> int:
    """Compute raw (unclamped) max batch size for a given sequence length.

    Returns 0 when even a single sample exceeds the VRAM budget.
    Used by warmup logic to skip shapes that genuinely don't fit.
    """
    budget_bytes = vram_budget_mb * 1024 * 1024
    per_sample_bytes = _VRAM_SAFETY_FACTOR * (
        seq_len * seq_len * _ATTN_COEFF + seq_len * _FFN_COEFF
    )
    if per_sample_bytes <= 0:
        return 32
    return int(budget_bytes / per_sample_bytes)


def max_batch_for_seq_len(seq_len: int, vram_budget_mb: float, is_gpu: bool = True) -> int:
    """Compute max safe batch size for a given sequence length and VRAM budget.

    Uses the transformer VRAM formula:
        total_bytes = safety × batch × (seq² × 288 + seq × 73728)

    This accounts for quadratic attention scaling — short sequences get large
    batches, long sequences get small batches, automatically.

    Args:
        seq_len: Max token count in the batch (determines padding).
        vram_budget_mb: Available VRAM/RAM in MB.
        is_gpu: Whether running on GPU (affects max cap).

    Returns:
        Safe batch size (clamped to [1, 32] for GPU, [1, 16] for CPU).
    """
    raw = _raw_batch_for_seq_len(seq_len, vram_budget_mb)
    max_cap = 32 if is_gpu else 16
    return max(1, min(raw, max_cap))


def compute_optimal_thread_counts(backend: str) -> tuple[int, int]:
    """Compute optimal ONNX Runtime thread counts.

    For GPU backends, returns (0, 0) to let ORT auto-tune — its internal
    heuristics are better than manual overrides for GPU execution providers.
    For CPU-only, sets explicit thread counts targeting ~85% core utilization.

    Args:
        backend: Active backend string.

    Returns:
        (intra_op_threads, inter_op_threads) tuple.
        0 means "let ORT decide" (recommended for GPU).
    """
    cores = detect_cpu_cores()
    is_gpu = "CUDA" in backend or "DirectML" in backend

    if is_gpu:
        # Let ORT auto-tune — overriding causes contention with GPU scheduling
        return 0, 0
    else:
        # CPU-only: maximize intra-op parallelism for matrix ops.
        usable = max(2, int(cores * TARGET_UTILIZATION))
        intra = usable
        inter = max(1, usable // 4)
        return intra, inter


def compute_optimal_embedding_batch_size() -> int:
    """Compute optimal ChromaDB embedding batch size (chunks per DB write).

    Based on available system RAM. Larger batches reduce DB round-trips
    but use more memory for the document + metadata lists.

    Returns:
        Optimal batch size (clamped to [50, 500]).
    """
    ram_mb = detect_system_ram_mb()
    # ~1 KB per chunk (embedding text + metadata) is a reasonable estimate.
    # At 100 chunks = ~100 KB, at 500 chunks = ~500 KB — both trivial.
    # Scale with RAM but the real bottleneck is embedding speed, not DB writes.
    if ram_mb >= 32768:  # 32 GB+
        return 500
    elif ram_mb >= 16384:  # 16 GB+
        return 250
    elif ram_mb >= 8192:  # 8 GB+
        return 150
    else:
        return 50


class HardwareProfile:
    """Detected hardware profile with optimal configuration values."""

    def __init__(self):
        self.gpu_vram_mb = detect_gpu_vram_mb()
        self.gpu_free_vram_mb = detect_gpu_free_vram_mb()
        self.cpu_cores = detect_cpu_cores()
        self.system_ram_mb = detect_system_ram_mb()

        # These are computed after backend detection (set by embedder)
        self.vram_budget_mb: float = 1500.0  # default until backend known
        self.is_gpu: bool = False
        self.intra_op_threads: int = 4
        self.inter_op_threads: int = 1
        self.embedding_batch_size: int = compute_optimal_embedding_batch_size()

    # How often to re-probe free VRAM (seconds). Balances responsiveness
    # to changing GPU load vs nvidia-smi subprocess overhead (~10ms each).
    _VRAM_REFRESH_INTERVAL = 30.0

    def configure_for_backend(self, backend: str) -> None:
        """Recompute optimal values once the ONNX backend is known."""
        import time
        self.backend = backend
        self.is_gpu = "CUDA" in backend or "DirectML" in backend
        self._is_directml = "DirectML" in backend
        self._vram_last_checked = 0.0  # force immediate refresh
        self._refresh_vram_budget()
        self.intra_op_threads, self.inter_op_threads = compute_optimal_thread_counts(backend)

    def _refresh_vram_budget(self) -> None:
        """Re-check free VRAM and update budget, at most every _VRAM_REFRESH_INTERVAL seconds."""
        import time
        now = time.monotonic()
        if now - getattr(self, "_vram_last_checked", 0.0) < self._VRAM_REFRESH_INTERVAL:
            return
        self._vram_last_checked = now

        if self.is_gpu:
            fresh_free = detect_gpu_free_vram_mb()
            if fresh_free is not None:
                self.gpu_free_vram_mb = fresh_free
        self.vram_budget_mb = compute_vram_budget_mb(
            self.gpu_vram_mb, self.gpu_free_vram_mb,
            getattr(self, "backend", "ONNX + CPU"))

    def max_batch_for_seq(self, seq_len: int) -> int:
        """Compute safe batch size for a given sequence length.

        Periodically re-checks free VRAM to adapt to changing GPU load.
        DirectML batches are capped lower because WDDM memory pools retain
        allocations per shape and never release during the session lifetime.
        """
        self._refresh_vram_budget()
        batch = max_batch_for_seq_len(seq_len, self.vram_budget_mb, self.is_gpu)
        if getattr(self, "_is_directml", False):
            batch = min(batch, DIRECTML_MAX_BATCH)
        return batch

    def seq_fits_in_vram(self, seq_len: int) -> bool:
        """Check if even a single sample at this sequence length fits in the VRAM budget.

        Used by warmup to skip shapes that would blow past VRAM and cause the
        GPU memory allocator to grab everything without releasing it.
        """
        self._refresh_vram_budget()
        return _raw_batch_for_seq_len(seq_len, self.vram_budget_mb) >= 1

    def summary(self) -> str:
        """Human-readable summary for logging and status reporting."""
        parts = []
        if self.gpu_vram_mb is not None:
            free = f", {self.gpu_free_vram_mb} MB free" if self.gpu_free_vram_mb else ""
            parts.append(f"GPU: {self.gpu_vram_mb} MB VRAM{free}")
        else:
            parts.append("GPU: not detected")
        parts.append(f"CPU: {self.cpu_cores} cores")
        parts.append(f"RAM: {self.system_ram_mb} MB")
        # Show adaptive batch examples at typical sequence lengths
        b256 = self.max_batch_for_seq(256)
        b512 = self.max_batch_for_seq(512)
        b1024 = self.max_batch_for_seq(1024)
        parts.append(f"ONNX batch: adaptive (256tok={b256}, 512tok={b512}, 1k={b1024})")
        parts.append(f"VRAM budget: {self.vram_budget_mb:.0f} MB")
        if self.intra_op_threads == 0:
            parts.append("Threads: auto (ORT-managed)")
        else:
            parts.append(f"Threads: intra={self.intra_op_threads}, inter={self.inter_op_threads}")
        parts.append(f"Embedding batch: {self.embedding_batch_size}")
        return " | ".join(parts)


# Singleton — initialized once at import time (detection is fast)
_profile: HardwareProfile | None = None


def get_hardware_profile() -> HardwareProfile:
    """Get or create the singleton HardwareProfile."""
    global _profile
    if _profile is None:
        _profile = HardwareProfile()
        logger.info("Hardware detected: %s", _profile.summary())
    return _profile
