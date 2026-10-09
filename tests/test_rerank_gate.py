"""Tests for the relevance gate in search: what it drops, what it keeps, the
one-line miss, exact-string hits under the same gate, and that a missing or
broken gate leaves search exactly as it was."""

import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tests.test_search import SearchTestCase, _row  # noqa: E402


class FakeGate:
    """Scores a candidate by the first keyword of its content found in `scores`."""

    def __init__(self, scores, fail=False):
        self.scores = scores
        self.fail = fail
        self.calls = []

    def score(self, query, texts):
        self.calls.append((query, list(texts)))
        if self.fail:
            raise RuntimeError("session died")
        out = []
        for t in texts:
            content = t.split("\n", 1)[1]
            out.append(next((s for k, s in self.scores.items() if k in content), 0.0))
        return out


class GateTestCase(SearchTestCase):
    gate = None

    def setUp(self):
        super().setUp()
        from src.indexer import reranker
        reranker.set_for_tests(self.gate)

    def tearDown(self):
        from src.indexer import reranker
        reranker.set_for_tests(None)
        super().tearDown()

    def ids(self, **kwargs):
        return [r["observation_id"] for r in self.search_json(**kwargs)["results"]]


class TestGateKeepsVectorOrder(GateTestCase):
    rows = [
        _row("o1", "e1", 0.10, "alpha fact"),
        _row("o2", "e1", 0.20, "beta fact"),
        _row("o3", "e2", 0.30, "gamma fact"),
        _row("o4", "e2", 0.40, "delta fact"),
        _row("o5", "e3", 1.20, "epsilon fact"),
    ]
    gate = FakeGate({"alpha": 0.05, "beta": 0.9, "gamma": 0.2, "delta": 0.01, "epsilon": 0.6})

    def test_drops_under_min_and_keeps_vector_order(self):
        # gamma (0.2) outranks epsilon (0.6) because order stays the vector's.
        self.assertEqual(self.ids(vault="test"), ["o2", "o3", "o5"])

    def test_low_band_hit_survives_when_relevant(self):
        payload = self.search_json(vault="test")
        eps = [r for r in payload["results"] if r["observation_id"] == "o5"][0]
        self.assertEqual(eps["confidence"], "LOW")
        self.assertAlmostEqual(eps["rerank"], 0.6)

    def test_caps_at_n_results(self):
        self.assertEqual(self.ids(vault="test", n_results=2), ["o2", "o3"])

    def test_gate_reads_entity_context(self):
        self.search_json(vault="test")
        query, texts = self.gate.calls[-1]
        self.assertEqual(query, "a query")
        self.assertTrue(texts[0].startswith("technology: E1\n"))

    def test_vector_pass_fetches_a_full_pool(self):
        from src.indexer.reranker import RERANK_POOL
        self.search_json(vault="test", n_results=1)
        self.assertGreaterEqual(self.collections["test"].queries[0]["n_results"], RERANK_POOL)


class TestGateMiss(GateTestCase):
    rows = [
        _row("o1", "e1", 0.90, "printer brands"),
        _row("o2", "e2", 1.10, "sleep schedule"),
        _row("o3", "e3", 1.50, "bird photos"),
    ]
    gate = FakeGate({})

    def test_text_miss_is_one_line(self):
        from src.tools.search import search_memory
        self.assertEqual(search_memory("my favourite food", vault="test"),
                         "No memory matches 'my favourite food'.")

    def test_json_miss_is_empty(self):
        payload = self.search_json(vault="test")
        self.assertEqual(payload["results"], [])
        self.assertEqual(payload["returned"], 0)

    def test_no_min_three_padding(self):
        self.assertEqual(self.ids(vault="test"), [])


class TestBrokenGateFallsBack(GateTestCase):
    rows = [
        _row("o1", "e1", 1.20, "low one"),
        _row("o2", "e2", 1.50, "noise one"),
        _row("o3", "e3", 1.60, "noise two"),
    ]
    gate = FakeGate({}, fail=True)

    def test_min_three_rule_returns(self):
        # Same as an ungated search: one LOW hit, padded to three.
        self.assertEqual(self.ids(vault="test"), ["o1", "o2", "o3"])


class TestNoGateUnchanged(GateTestCase):
    rows = TestBrokenGateFallsBack.rows
    gate = None

    def test_min_three_rule_returns(self):
        payload = self.search_json(vault="test")
        self.assertEqual([r["observation_id"] for r in payload["results"]], ["o1", "o2", "o3"])
        self.assertTrue(all(r["rerank"] is None for r in payload["results"]))


class TestKeywordHitsAreGated(GateTestCase):
    rows = [
        _row("o1", "e1", 1.20, "deploy notes for the hub"),
        _row("o2", "e2", 1.30, "unrelated chatter"),
    ]
    gate = FakeGate({"deploy notes": 0.7, "kubeconfig": 0.8, "orphan": 0.01})

    def setUp(self):
        super().setUp()
        hits = [
            {"observation_id": "k1", "entity_id": "e9", "entity_name": "E9",
             "entity_type": "technology", "content": "kubeconfig lives in C:/keys",
             "source": "", "vault": "test", "distance": None, "graph_boosted": False,
             "superseded": False, "keyword_match": True},
            {"observation_id": "k2", "entity_id": "e9", "entity_name": "E9",
             "entity_type": "technology", "content": "orphan path C:/keys/old",
             "source": "", "vault": "test", "distance": None, "graph_boosted": False,
             "superseded": False, "keyword_match": True},
        ]
        p = patch("src.tools.search._keyword_search", return_value=hits)
        p.start()
        self.addCleanup(p.stop)

    def test_keyword_hits_pass_the_same_gate(self):
        self.assertEqual(self.ids(vault="test"), ["o1", "k1"])

    def test_keyword_hits_count_toward_n(self):
        self.assertEqual(self.ids(vault="test", n_results=1), ["o1"])


class TestReranker(unittest.TestCase):
    def test_missing_model_without_fetch_is_reported(self):
        from pathlib import Path
        from src.indexer import reranker
        with patch.object(reranker, "RERANK_ONNX_DIR", Path("Z:/nowhere")), \
                patch.object(reranker, "model_present", return_value=False), \
                patch.object(reranker, "_reranker", None), \
                patch.object(reranker, "_load_error", None),                 patch.object(reranker, "RERANK_MODE", "on"):
            self.assertIsNone(reranker.load(fetch=False))
            self.assertIn("FileNotFoundError", reranker.status()["error"])
            self.assertIsNone(reranker.get())

    def test_off_never_loads(self):
        from src.indexer import reranker
        with patch.object(reranker, "RERANK_MODE", "off"), \
                patch.object(reranker, "_reranker", None), \
                patch.object(reranker, "download") as dl:
            self.assertIsNone(reranker.load())
            dl.assert_not_called()


if __name__ == "__main__":
    unittest.main()
