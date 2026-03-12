"""Minimal hardware detection for CPU-only operation.

Memory-index embeds one observation at a time (~10ms), so GPU acceleration
adds complexity with no benefit. This is a stripped-down version of
code-index's hardware.py.
"""

import logging
import os

logger = logging.getLogger(__name__)


def detect_cpu_cores() -> int:
    """Detect number of logical CPU cores."""
    return os.cpu_count() or 4


def get_cpu_thread_counts() -> tuple[int, int]:
    """Compute optimal ONNX Runtime CPU thread counts.

    Returns:
        (intra_op_threads, inter_op_threads) tuple.
    """
    cores = detect_cpu_cores()
    usable = max(2, int(cores * 0.5))
    intra = usable
    inter = max(1, usable // 4)
    return intra, inter
