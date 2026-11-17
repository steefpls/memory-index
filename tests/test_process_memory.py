"""GET /health's `memory` block: cached, cheap, and never in the way."""
import sys
import threading
import time
import unittest
from unittest import mock

from src.process_memory import ProcessMemory


class _FakeGpu:
    def __init__(self, committed=620.5):
        self.reads = 0
        self.committed = committed

    def read(self):
        self.reads += 1
        return {"dedicated_mb": 564.5, "shared_mb": 56.0, "committed_mb": self.committed}


def _wait_for(pm, state, timeout=5.0):
    end = time.monotonic() + timeout
    while pm._gpu_state != state and time.monotonic() < end:
        time.sleep(0.01)
    return pm._gpu_state


class TestProcessMemory(unittest.TestCase):
    def test_snapshot_has_the_process_figures(self):
        snap = ProcessMemory(gpu_factory=None).snapshot()
        self.assertGreater(snap["private_mb"], 0)
        self.assertGreater(snap["working_set_mb"], 0)
        self.assertGreaterEqual(snap["threads"], 1)
        self.assertEqual(set(snap["arena_releases"]), {"embedder", "reranker"})

    def test_no_gpu_counters_means_null_gpu_and_host(self):
        pm = ProcessMemory(gpu_factory=None)
        pm._gpu_state = "unavailable"
        snap = pm.snapshot()
        self.assertIsNone(snap["gpu"])
        self.assertIsNone(snap["host_mb"])

    def test_gpu_opens_in_the_background_then_shows_with_host_mb(self):
        gate = threading.Event()
        gpu = _FakeGpu()

        def slow_open():
            gate.wait(5)
            return gpu

        pm = ProcessMemory(cache_s=60, gpu_factory=slow_open)
        first = pm.snapshot()  # must not wait for the slow open
        self.assertIsNone(first["gpu"])
        gate.set()
        self.assertEqual(_wait_for(pm, "ready"), "ready")
        # Opening clears the cache, so the next call shows the GPU at once.
        snap = pm.snapshot()
        self.assertEqual(snap["gpu"]["committed_mb"], 620.5)
        self.assertAlmostEqual(snap["host_mb"], snap["private_mb"] - 620.5, places=1)

    def test_a_failed_open_leaves_gpu_null_and_is_not_retried(self):
        calls = []

        def broken():
            calls.append(1)
            raise OSError("no counter")

        pm = ProcessMemory(cache_s=0, gpu_factory=broken)
        pm.snapshot()
        self.assertEqual(_wait_for(pm, "unavailable"), "unavailable")
        self.assertIsNone(pm.snapshot()["gpu"])
        self.assertIsNone(pm.snapshot()["gpu"])
        self.assertEqual(len(calls), 1)

    def test_snapshots_are_cached(self):
        gpu = _FakeGpu()
        pm = ProcessMemory(cache_s=60, gpu_factory=lambda: gpu)
        pm._gpu, pm._gpu_state = gpu, "ready"
        a = pm.snapshot()
        b = pm.snapshot()
        self.assertIs(a, b)
        self.assertEqual(gpu.reads, 1)

    def test_a_gpu_read_error_only_nulls_gpu(self):
        class Bad:
            def read(self):
                raise OSError("collect failed")

        pm = ProcessMemory(cache_s=0, gpu_factory=lambda: Bad())
        pm._gpu, pm._gpu_state = Bad(), "ready"
        snap = pm.snapshot()
        self.assertIsNone(snap["gpu"])
        self.assertGreater(snap["private_mb"], 0)

    def test_unreadable_process_gives_none(self):
        pm = ProcessMemory(cache_s=0, gpu_factory=None)
        with mock.patch("psutil.Process", side_effect=RuntimeError("denied")):
            self.assertIsNone(pm.snapshot())

    def test_arena_release_counts_come_from_the_loaded_sessions(self):
        from src.indexer import embedder, reranker
        fn = mock.Mock()
        fn._releaser.releases = 7
        with mock.patch.object(embedder, "_embedding_fn", fn), \
                mock.patch.object(reranker, "get", return_value=None):
            snap = ProcessMemory(cache_s=0, gpu_factory=None).snapshot()
        self.assertEqual(snap["arena_releases"], {"embedder": 7, "reranker": None})

    @unittest.skipUnless(sys.platform == "win32", "Windows GPU counters")
    def test_real_windows_counters_read_this_process(self):
        from src.process_memory import _GpuCounters
        try:
            gpu = _GpuCounters(__import__("os").getpid())
        except OSError as e:  # a box without the GPU counter set
            self.skipTest(f"no GPU Process Memory counters: {e}")
        out = gpu.read()
        self.assertEqual(set(out), {"dedicated_mb", "shared_mb", "committed_mb"})
        for v in out.values():
            self.assertGreaterEqual(v, 0)


if __name__ == "__main__":
    unittest.main()
