"""This process's memory figures, for GET /health's `memory` block.

The hub reads /health on a timer, so everything here is cheap and cached:
one snapshot is reused for CACHE_S seconds, and the slow parts happen once
(opening the Windows GPU counter query costs ~0.6 s, so it opens in a
background thread and `gpu` is null until it is ready; after that a read
takes well under a millisecond).

Why the GPU figure matters (measured 2026-11-17, see the memory-index leak
investigation): on Windows the CUDA driver backs every device allocation
with process commit, so `private_mb` rises and falls with the ONNX CUDA
arenas, from ~2.8 GB idle to ~4.6 GB in a burst. `host_mb` (private minus
GPU commit) is the line that stays flat, about 2.15 GB; a leak shows as
that floor creeping, not as a high private reading mid-traffic.
"""
from __future__ import annotations

import logging
import os
import sys
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

CACHE_S = 5.0
_MB = 2**20


def _mb(n: float) -> float:
    return round(n / _MB, 1)


class _GpuCounters:
    """Per-process GPU memory from Windows' "GPU Process Memory" counters.

    Instances are named pid_<pid>_luid_<adapter>_phys_<n>; the wildcard query
    re-expands on every collect, so memory the process gets after the query
    opened still shows up. Values are summed over adapters.
    """

    NAMES = {"dedicated_mb": "Dedicated Usage", "shared_mb": "Shared Usage",
             "committed_mb": "Total Committed"}
    _PDH_FMT_LARGE = 0x400
    _PDH_MORE_DATA = 0x800007D2
    _PDH_NO_DATA = 0x800007D5

    def __init__(self, pid: int):
        import ctypes
        from ctypes import wintypes as w

        self._ct, self._w = ctypes, w
        pdh = ctypes.WinDLL("pdh")
        pdh.PdhOpenQueryW.argtypes = [w.LPCWSTR, ctypes.c_size_t, ctypes.POINTER(w.HANDLE)]
        pdh.PdhAddEnglishCounterW.argtypes = [w.HANDLE, w.LPCWSTR, ctypes.c_size_t,
                                              ctypes.POINTER(w.HANDLE)]
        pdh.PdhCollectQueryData.argtypes = [w.HANDLE]
        pdh.PdhGetFormattedCounterArrayW.argtypes = [
            w.HANDLE, w.DWORD, ctypes.POINTER(w.DWORD), ctypes.POINTER(w.DWORD), ctypes.c_void_p]
        for f in (pdh.PdhOpenQueryW, pdh.PdhAddEnglishCounterW, pdh.PdhCollectQueryData,
                  pdh.PdhGetFormattedCounterArrayW):
            f.restype = w.DWORD

        class _Fmt(ctypes.Structure):
            _fields_ = [("CStatus", w.DWORD), ("largeValue", ctypes.c_longlong)]

        class _Item(ctypes.Structure):
            _fields_ = [("szName", w.LPWSTR), ("FmtValue", _Fmt)]

        self._pdh, self._Item = pdh, _Item
        self._query = w.HANDLE()
        st = pdh.PdhOpenQueryW(None, 0, ctypes.byref(self._query))
        if st:
            raise OSError(f"PdhOpenQueryW {st:#x}")
        self._counters: dict[str, Any] = {}
        for key, name in self.NAMES.items():
            c = w.HANDLE()
            path = rf"\GPU Process Memory(pid_{pid}_*)\{name}"
            st = pdh.PdhAddEnglishCounterW(self._query, path, 0, ctypes.byref(c))
            if st:
                raise OSError(f"PdhAddEnglishCounterW {name} {st:#x}")
            self._counters[key] = c

    def read(self) -> dict[str, float]:
        ct, w, pdh = self._ct, self._w, self._pdh
        st = pdh.PdhCollectQueryData(self._query)
        if st == self._PDH_NO_DATA:  # no instance at all: no GPU memory held
            return {key: 0.0 for key in self._counters}
        if st:
            raise OSError(f"PdhCollectQueryData {st:#x}")
        out: dict[str, float] = {}
        for key, c in self._counters.items():
            size, n = w.DWORD(0), w.DWORD(0)
            st = pdh.PdhGetFormattedCounterArrayW(c, self._PDH_FMT_LARGE, ct.byref(size),
                                                  ct.byref(n), None)
            if st == self._PDH_NO_DATA:  # no instance: this process holds no GPU memory
                out[key] = 0.0
                continue
            if st != self._PDH_MORE_DATA:
                raise OSError(f"PdhGetFormattedCounterArrayW {key} {st:#x}")
            buf = ct.create_string_buffer(size.value)
            st = pdh.PdhGetFormattedCounterArrayW(c, self._PDH_FMT_LARGE, ct.byref(size),
                                                  ct.byref(n), buf)
            if st:
                raise OSError(f"PdhGetFormattedCounterArrayW {key} {st:#x}")
            items = ct.cast(buf, ct.POINTER(self._Item))
            out[key] = _mb(sum(items[i].FmtValue.largeValue for i in range(n.value)
                               if items[i].FmtValue.CStatus in (0, 1)))
        return out


class ProcessMemory:
    """Cached snapshot source; one per process (see `snapshot`)."""

    def __init__(self, cache_s: float = CACHE_S, gpu_factory=None):
        self._cache_s = cache_s
        self._lock = threading.Lock()
        self._at = 0.0
        self._last: dict | None = None
        self._gpu: Any = None
        self._gpu_state = "unopened"  # unopened | opening | ready | unavailable
        if gpu_factory is None and sys.platform == "win32":
            gpu_factory = lambda: _GpuCounters(os.getpid())  # noqa: E731
        self._gpu_factory = gpu_factory
        if gpu_factory is None:
            self._gpu_state = "unavailable"

    def _open_gpu(self) -> None:
        try:
            gpu = self._gpu_factory()
            with self._lock:
                self._gpu, self._gpu_state = gpu, "ready"
                self._at = 0.0  # the next /health shows it, not a cached null
        except Exception as e:  # noqa: BLE001 — no counters: report gpu as null
            logger.info("GPU memory counters unavailable: %s", e)
            with self._lock:
                self._gpu_state = "unavailable"

    def _read_gpu(self) -> dict | None:
        if self._gpu_state == "unopened":
            self._gpu_state = "opening"
            threading.Thread(target=self._open_gpu, name="gpu-mem-counters",
                             daemon=True).start()
            return None
        if self._gpu_state != "ready":
            return None
        try:
            return self._gpu.read()
        except Exception as e:  # noqa: BLE001
            logger.debug("GPU memory read failed: %s", e)
            return None

    def _read(self) -> dict:
        import psutil

        p = psutil.Process()
        mi = p.memory_info()
        private = getattr(mi, "private", None) or mi.vms
        out: dict[str, Any] = {
            "private_mb": _mb(private),
            "working_set_mb": _mb(mi.rss),
            "threads": p.num_threads(),
        }
        peak = getattr(mi, "peak_pagefile", None)
        if peak is not None:
            out["peak_private_mb"] = _mb(peak)
        handles = getattr(p, "num_handles", None)
        if handles is not None:
            out["handles"] = handles()
        gpu = self._read_gpu()
        out["gpu"] = gpu
        out["host_mb"] = (round(out["private_mb"] - gpu["committed_mb"], 1)
                          if gpu and "committed_mb" in gpu else None)
        out["arena_releases"] = _arena_releases()
        return out

    def snapshot(self) -> dict | None:
        """The figures, at most CACHE_S old; None when they can't be read."""
        with self._lock:
            now = time.monotonic()
            if self._last is not None and now - self._at < self._cache_s:
                return self._last
            try:
                self._last = self._read()
            except Exception as e:  # noqa: BLE001 — /health must answer anyway
                logger.debug("process memory unreadable: %s", e)
                self._last = None
            self._at = now
            return self._last


def _arena_releases() -> dict:
    """How many times each CUDA session's idle timer handed its arena back
    (null when the session isn't loaded or runs without a releaser)."""
    from src.indexer import embedder, reranker

    def count(obj: Any) -> int | None:
        return getattr(getattr(obj, "_releaser", None), "releases", None)

    return {"embedder": count(embedder._embedding_fn), "reranker": count(reranker.get())}


_default = ProcessMemory()


def snapshot() -> dict | None:
    return _default.snapshot()
