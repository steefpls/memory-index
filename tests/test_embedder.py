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
        emb._embedding_fn = None
        emb._active_backend = "not initialized"

    def tearDown(self):
        self.emb._embedding_fn = self._saved_fn
        self.emb._active_backend = self._saved_backend

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


if __name__ == "__main__":
    unittest.main()
