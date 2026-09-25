"""Tests for the embedder/Chroma singletons.

Construction is expensive (a full EmbeddingGemma-300m ONNX session, hundreds of
MB resident), so it must happen exactly once no matter how many threads race
for it. The real model is never loaded here — GemmaEmbedder is replaced by a
deliberately slow fake that only counts constructions.
"""

import os
import sys
import threading
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestEmbeddingSingleton(unittest.TestCase):
    def setUp(self):
        import src.indexer.embedder as emb
        self.emb = emb
        self._saved_fn = emb._embedding_fn
        self._saved_backend = emb._active_backend
        self._saved_error = emb._load_error
        emb._embedding_fn = None
        emb._active_backend = "not initialized"

    def tearDown(self):
        self.emb._embedding_fn = self._saved_fn
        self.emb._active_backend = self._saved_backend
        self.emb._load_error = self._saved_error

    def _slow_fake(self, counter):
        class FakeEmbedder:
            def __init__(self):
                counter.append(1)
                # Wide enough that an unsynchronized check-then-set is
                # guaranteed to let a second caller in.
                time.sleep(0.2)
                self.backend = "fake"

            def close(self):
                self.backend = "released"

        return FakeEmbedder

    def test_concurrent_first_callers_build_exactly_one_embedder(self):
        """The bug: store writes, calibration, the auto-librarian thread and
        search all call get_embedding_function() directly. Two of them racing
        each built a full ONNX session; one was leaked unreferenced."""
        built = []
        results = []
        barrier = threading.Barrier(8)
        lock = threading.Lock()

        def worker():
            barrier.wait()
            fn = self.emb.get_embedding_function()
            with lock:
                results.append(fn)

        with patch.object(self.emb, "GemmaEmbedder", self._slow_fake(built)):
            threads = [threading.Thread(target=worker) for _ in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(30)

        self.assertEqual(len(built), 1, "more than one embedder was constructed")
        self.assertEqual(len(results), 8)
        # Every caller got the same object — nothing was leaked.
        self.assertEqual(len({id(r) for r in results}), 1)

    def test_active_backend_is_set_before_the_singleton_is_published(self):
        """A caller that sees _embedding_fn must also see the real backend
        string, never the 'not initialized' placeholder."""
        built = []
        with patch.object(self.emb, "GemmaEmbedder", self._slow_fake(built)):
            self.emb.get_embedding_function()
        self.assertEqual(self.emb.get_active_backend(), "fake")

    def test_release_is_safe_and_allows_rebuild(self):
        built = []
        with patch.object(self.emb, "GemmaEmbedder", self._slow_fake(built)):
            self.emb.get_embedding_function()
            self.emb.release_embedding_function()
            self.assertEqual(self.emb.get_active_backend(), "not initialized")
            self.emb.get_embedding_function()
        self.assertEqual(len(built), 2)


    def test_a_failed_load_is_remembered_until_one_succeeds(self):
        """/health reads get_load_error(): a model that won't load must show
        there, and a later successful load must clear it."""
        class Broken:
            def __init__(self):
                raise AttributeError("module 'onnxruntime' has no attribute 'SessionOptions'")
        with patch.object(self.emb, "GemmaEmbedder", Broken):
            with self.assertRaises(AttributeError):
                self.emb.get_embedding_function()
        self.assertIn("SessionOptions", self.emb.get_load_error())
        with patch.object(self.emb, "GemmaEmbedder", self._slow_fake([])):
            self.emb.get_embedding_function()
        self.assertIsNone(self.emb.get_load_error())


class TestChromaClientSingleton(unittest.TestCase):
    def setUp(self):
        import src.indexer.embedder as emb
        self.emb = emb
        self._saved = emb._client
        emb._client = None

    def tearDown(self):
        self.emb._client = self._saved

    def test_concurrent_callers_build_one_client(self):
        built = []
        results = []
        barrier = threading.Barrier(6)
        lock = threading.Lock()

        def slow_client(path=None):
            built.append(1)
            time.sleep(0.2)
            return object()

        def worker():
            barrier.wait()
            client = self.emb.get_chroma_client()
            with lock:
                results.append(client)

        with patch.object(self.emb.chromadb, "PersistentClient",
                          side_effect=slow_client):
            threads = [threading.Thread(target=worker) for _ in range(6)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(30)

        self.assertEqual(len(built), 1)
        self.assertEqual(len({id(r) for r in results}), 1)


class TestOnnxBatching(unittest.TestCase):
    """Large inputs must be split into bounded ONNX batches.

    Regression: a rename re-embed passed ~900 texts in one session run and
    OOM-killed the daemon with no traceback.
    """

    def _embedder_with_fake_session(self):
        import numpy as np
        import src.indexer.embedder as emb_mod
        e = emb_mod.GemmaEmbedder.__new__(emb_mod.GemmaEmbedder)
        seen = []

        class FakeSession:
            def run(self, names, feed, run_options=None):
                n = feed["input_ids"].shape[0]
                seen.append(n)
                return [np.zeros((n, 768), dtype=np.float32)]

        class FakeTok:
            def __call__(self, texts, **kw):
                n = len(texts)
                return {"input_ids": np.zeros((n, 8), dtype=np.int64),
                        "attention_mask": np.ones((n, 8), dtype=np.int64)}

        e._ort_session = FakeSession()
        e._run_opts = None
        e._tokenizer = FakeTok()
        e._pt_model = None
        return e, seen

    def test_large_input_is_split_into_bounded_batches(self):
        e, seen = self._embedder_with_fake_session()
        out = e._onnx_embed([f"text {i}" for i in range(70)])
        self.assertEqual(len(out), 70)
        self.assertTrue(seen, "session was never called")
        self.assertTrue(all(n <= 32 for n in seen), seen)
        self.assertEqual(sum(seen), 70)

    def test_small_input_takes_single_batch(self):
        e, seen = self._embedder_with_fake_session()
        out = e._onnx_embed(["a", "b", "c"])
        self.assertEqual(len(out), 3)
        self.assertEqual(seen, [3])

    def test_empty_input(self):
        e, seen = self._embedder_with_fake_session()
        self.assertEqual(e._onnx_embed([]), [])
        self.assertEqual(seen, [])

    def test_cell_budget_isolates_long_texts_and_preserves_order(self):
        """One monster text must not drag its whole chunk down with it
        (padding expands every row to the longest): it gets a small chunk,
        and returned embeddings still align with input order."""
        import src.indexer.embedder as emb_mod
        e, _ = self._embedder_with_fake_session()

        lengths = [50] * 5 + [2000] + [50] * 64
        e._embed_lengths = lambda texts: lengths[:len(texts)]

        batch_rows = []

        def fake_batch(chunk):
            batch_rows.append(len(chunk))
            return [[float(t[1:])] for t in chunk]

        e._onnx_embed_batch = fake_batch
        out = e._onnx_embed([f"t{i}" for i in range(70)])

        # Order preserved: out[i] is the embedding of input i.
        self.assertEqual([row[0] for row in out], [float(i) for i in range(70)])
        # The 2000-token row sits in a small chunk, not a 32-row one.
        for b, chunk_size in enumerate(batch_rows):
            self.assertLessEqual(chunk_size, 32, f"chunk {b} too big")
        long_chunk = [n for n in batch_rows if n <= 4]
        self.assertTrue(long_chunk, f"no small chunk isolated the long text: {batch_rows}")


class TestProviderChoice(unittest.TestCase):
    """providers_for is the whole device policy, pure so it runs without a GPU."""

    def setUp(self):
        from src.indexer.embedder import providers_for
        self.pf = providers_for
        self.cuda = "CUDAExecutionProvider"
        self.cpu = "CPUExecutionProvider"

    def _names(self, providers):
        return [p[0] if isinstance(p, tuple) else p for p in providers]

    def test_auto_takes_cuda_when_the_build_offers_it(self):
        providers, err = self.pf("auto", [self.cuda, self.cpu])
        self.assertEqual(self._names(providers), [self.cuda, self.cpu])
        self.assertIsNone(err)
        # The arena grows by what a run needs: the card is shared with Whisper.
        self.assertEqual(providers[0][1]["arena_extend_strategy"], "kSameAsRequested")

    def test_auto_is_quietly_cpu_on_a_plain_build(self):
        providers, err = self.pf("auto", [self.cpu])
        self.assertEqual(providers, [self.cpu])
        self.assertIsNone(err)

    def test_cuda_demanded_but_missing_is_an_error_not_a_silent_fallback(self):
        providers, err = self.pf("cuda", ["AzureExecutionProvider", self.cpu])
        self.assertEqual(providers, [self.cpu])
        self.assertIn("no CUDA provider", err)
        self.assertIn("cuda", err)  # names the dependency group to install

    def test_cpu_never_touches_the_card(self):
        providers, err = self.pf("cpu", [self.cuda, self.cpu])
        self.assertEqual(providers, [self.cpu])
        self.assertIsNone(err)

    def test_typo_is_reported_and_runs_on_cpu(self):
        providers, err = self.pf("gpu", [self.cuda, self.cpu])
        self.assertEqual(providers, [self.cpu])
        self.assertIn("gpu", err)


class TestArenaRelease(unittest.TestCase):
    """A CUDA session gives its arena back once runs go quiet, not on every run.

    Regression 1: kSameAsRequested alone let the arena keep its peak for the
    life of the process; the live daemon held 2.9 GB of the shared GTX 1070
    for a ~400 MiB model (2026-09-25).
    Regression 2: shrinking on every run (77a664c) fixed that but doubled a
    single query's latency, 72 -> 138 ms, because each run reallocated its
    ~1.6 GB of working memory.
    """
    KEY = "memory.enable_memory_arena_shrinkage"

    def setUp(self):
        import src.indexer.embedder as emb
        self.emb = emb
        self._saved_after = emb.ARENA_RELEASE_SECONDS
        self._embedders = []

    def tearDown(self):
        for e in self._embedders:
            e.close()
        self.emb.ARENA_RELEASE_SECONDS = self._saved_after

    def test_only_a_shrink_run_on_cuda_gets_the_option(self):
        self.assertEqual(self.emb.run_config_for("CUDAExecutionProvider", shrink=True),
                         {self.KEY: "gpu:0"})
        self.assertEqual(self.emb.run_config_for("CUDAExecutionProvider"), {})
        self.assertEqual(self.emb.run_config_for("CPUExecutionProvider", shrink=True), {})

    def _fake_session(self, provider, calls, fail_batches=False, on_run=None):
        import numpy as np

        class FakeSession:
            def get_providers(self):
                return [provider, "CPUExecutionProvider"]

            def run(self, names, feed, run_options=None):
                n = feed["input_ids"].shape[0]
                calls.append((n, run_options))
                if on_run is not None:
                    on_run(run_options)
                if fail_batches and n > 1:
                    raise RuntimeError("batch too big")
                return [np.zeros((n, 768), dtype=np.float32)]

        return FakeSession()

    def _embedder(self, session, after=30.0):
        import numpy as np
        self.emb.ARENA_RELEASE_SECONDS = after
        e = self.emb.GemmaEmbedder.__new__(self.emb.GemmaEmbedder)

        class FakeTok:
            def __call__(self, texts, **kw):
                n = len(texts)
                return {"input_ids": np.zeros((n, 8), dtype=np.int64),
                        "attention_mask": np.ones((n, 8), dtype=np.int64)}

        e._ort_session = session
        e._run_opts = self.emb._run_options(session)
        e._releaser = None
        e._tokenizer = FakeTok()
        e._pt_model = None
        e._start_releaser(session)
        self._embedders.append(e)
        return e

    def _shrinks(self, run_options):
        return (run_options is not None
                and run_options.get_run_config_entry(self.KEY) == "gpu:0")

    def _wait_for(self, cond, timeout=3.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if cond():
                return True
            time.sleep(0.01)
        return cond()

    def test_normal_cuda_runs_keep_the_arena(self):
        """Batch, its one-at-a-time fallback and warmup must not shrink."""
        calls = []
        session = self._fake_session("CUDAExecutionProvider", calls,
                                     fail_batches=True)
        e = self._embedder(session)
        e.warmup()
        out = e._onnx_embed_batch(["a", "b", "c"])
        self.assertEqual(len(out), 3)
        # warmup, the failed batch, then three single-text retries
        self.assertEqual([n for n, _ in calls], [1, 3, 1, 1, 1])
        self.assertFalse(any(self._shrinks(ro) for _, ro in calls), calls)

    def test_load_time_smoke_run_releases_its_arena(self):
        calls = []
        session = self._fake_session("CUDAExecutionProvider", calls)
        e = self._embedder(session)
        e._smoke(session)
        self.assertTrue(self._shrinks(calls[-1][1]))

    def test_release_fires_once_after_quiet(self):
        calls = []
        e = self._embedder(self._fake_session("CUDAExecutionProvider", calls),
                           after=0.2)
        e._onnx_embed_batch(["a"])
        self.assertTrue(self._wait_for(lambda: e._releaser.releases == 1))
        self.assertTrue(self._shrinks(calls[-1][1]))
        time.sleep(0.4)  # nothing ran since, so nothing more to release
        self.assertEqual(e._releaser.releases, 1)
        e._onnx_embed_batch(["b"])  # a new burst arms it again
        self.assertTrue(self._wait_for(lambda: e._releaser.releases == 2))

    def test_no_release_during_a_burst(self):
        calls = []
        e = self._embedder(self._fake_session("CUDAExecutionProvider", calls),
                           after=0.3)
        end = time.monotonic() + 0.9
        while time.monotonic() < end:
            e._onnx_embed_batch(["q"])
            time.sleep(0.05)
        self.assertEqual(e._releaser.releases, 0)
        self.assertFalse(any(self._shrinks(ro) for _, ro in calls))
        self.assertTrue(self._wait_for(lambda: e._releaser.releases == 1))

    def test_release_run_holds_the_run_lock(self):
        held = []
        box = {}

        def on_run(run_options):
            if self._shrinks(run_options):
                lock = box["e"]._releaser.lock
                probe = threading.Thread(
                    target=lambda: held.append(not lock.acquire(timeout=0.05)))
                probe.start()
                probe.join()

        e = self._embedder(self._fake_session("CUDAExecutionProvider", [], on_run=on_run),
                           after=0.1)
        box["e"] = e
        e._onnx_embed_batch(["a"])
        self.assertTrue(self._wait_for(lambda: e._releaser.releases == 1))
        self.assertEqual(held, [True])

    def test_cpu_session_has_no_timer_and_no_run_options(self):
        calls = []
        session = self._fake_session("CPUExecutionProvider", calls,
                                     fail_batches=True)
        e = self._embedder(session, after=0.1)
        self.assertIsNone(e._releaser)
        e.warmup()
        e._smoke(session)
        e._onnx_embed_batch(["a", "b"])
        time.sleep(0.3)
        self.assertEqual([ro for _, ro in calls], [None] * len(calls))
        self.assertEqual(len(calls), 5)  # no release run appeared

    def test_zero_seconds_disables_the_release(self):
        e = self._embedder(self._fake_session("CUDAExecutionProvider", []), after=0)
        self.assertIsNone(e._releaser)

    def test_close_stops_the_timer_before_it_fires(self):
        calls = []
        e = self._embedder(self._fake_session("CUDAExecutionProvider", calls),
                           after=0.3)
        e._onnx_embed_batch(["a"])
        releaser = e._releaser
        e.close()
        self.assertFalse(releaser._thread.is_alive())
        self.assertIsNone(e._releaser)
        time.sleep(0.5)
        self.assertEqual(releaser.releases, 0)
        self.assertEqual(len(calls), 1)

    def test_release_seconds_from_env(self):
        with patch.dict(os.environ, {"MEMORY_INDEX_ARENA_RELEASE_SECONDS": "45"}):
            self.assertEqual(self.emb._release_seconds(), 45.0)
        with patch.dict(os.environ, {"MEMORY_INDEX_ARENA_RELEASE_SECONDS": "soon"}):
            self.assertEqual(self.emb._release_seconds(), 30.0)
        with patch.dict(os.environ, {"MEMORY_INDEX_ARENA_RELEASE_SECONDS": ""}):
            self.assertEqual(self.emb._release_seconds(), 30.0)


class TestEmbedDeviceReport(unittest.TestCase):
    """get_embed_device() is what memory_status and /health show."""

    def setUp(self):
        import src.indexer.embedder as emb
        self.emb = emb
        self._saved = emb._embedding_fn

    def tearDown(self):
        self.emb._embedding_fn = self._saved

    def test_nothing_loaded_yet(self):
        self.emb._embedding_fn = None
        d = self.emb.get_embed_device()
        self.assertEqual(d["requested"], self.emb.EMBED_DEVICE)
        self.assertIsNone(d["active"])
        self.assertIsNone(d["error"])

    def test_loaded_with_a_fallback_reports_the_error(self):
        class Fake:
            device = "cpu"
            device_error = "CUDA session failed, using CPU: CUDNN_STATUS_EXECUTION_FAILED"
        self.emb._embedding_fn = Fake()
        d = self.emb.get_embed_device()
        self.assertEqual(d["active"], "cpu")
        self.assertIn("CUDNN", d["error"])


if __name__ == "__main__":
    unittest.main()
