"""Tests for the retrieval eval harness (scripts/eval_search.py).

These are meta-tests: they check that the *measuring instrument* works — the
fixture vault builds through the real store APIs, the golden answer key is
internally consistent, the metric math is right, and the harness leaves no
global state behind. The recall floor asserted here is a plumbing regression
guard, NOT a statement about EmbeddingGemma's retrieval quality (fixture mode
runs a deterministic lexical stand-in embedder, not the real model).
"""

import importlib.util
import json
import os
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _load_eval_module():
    """scripts/ is not a package, so load the harness by path."""
    path = PROJECT_ROOT / "scripts" / "eval_search.py"
    spec = importlib.util.spec_from_file_location("eval_search", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["eval_search"] = module
    spec.loader.exec_module(module)
    return module


ev = _load_eval_module()

# Fixture mode must clear these or the harness is broken, not the search code.
MIN_RECALL_AT_5 = 0.90
MIN_MRR = 0.70


class TestGoldenDatasetIntegrity(unittest.TestCase):
    """The answer key has to be answerable — a typo here silently caps recall."""

    @classmethod
    def setUpClass(cls):
        cls.golden = ev.load_golden(ev.DEFAULT_GOLDEN)
        cls.corpus = {
            obs.strip()
            for spec in cls.golden["entities"]
            for obs in spec["observations"]
        }
        cls.entity_names = {spec["name"] for spec in cls.golden["entities"]}

    def test_corpus_size(self):
        self.assertGreaterEqual(len(self.golden["entities"]), 10)
        self.assertGreaterEqual(len(self.corpus), 40)

    def test_observations_are_unique(self):
        total = sum(len(s["observations"]) for s in self.golden["entities"])
        self.assertEqual(total, len(self.corpus),
                         "duplicate observation text makes 'expected' ambiguous")

    def test_query_ids_unique(self):
        ids = [q["id"] for q in self.golden["queries"]]
        self.assertEqual(len(ids), len(set(ids)))

    def test_at_least_25_queries(self):
        self.assertGreaterEqual(len(self.golden["queries"]), 25)

    def test_every_expectation_exists_in_the_corpus(self):
        for entry in self.golden["queries"]:
            self.assertTrue(
                entry.get("expect_contents") or entry.get("expect_entities"),
                f"{entry['id']} expects nothing",
            )
            for content in entry.get("expect_contents", []):
                self.assertIn(content.strip(), self.corpus,
                              f"{entry['id']} expects text that is not in the fixture")
            for name in entry.get("expect_entities", []):
                self.assertIn(name, self.entity_names,
                              f"{entry['id']} expects an entity that does not exist")

    def test_entities_declare_a_type_and_source(self):
        for spec in self.golden["entities"]:
            self.assertTrue(spec.get("type"))
            self.assertTrue(spec.get("source"), "source discipline applies to fixtures too")


class TestDeterministicEmbedder(unittest.TestCase):
    """Stable rankings require a stable embedder — this one must never drift."""

    def setUp(self):
        self.ef = ev.DeterministicEmbedder()

    def test_same_text_same_vector(self):
        a = self.ef.embed_queries(["fleet telemetry compression policy"])[0]
        b = self.ef.embed_queries(["fleet telemetry compression policy"])[0]
        self.assertEqual(a, b)

    def test_stable_across_instances(self):
        # Python's str hash is per-process randomized; this must not be.
        other = ev.DeterministicEmbedder()
        self.assertEqual(self.ef.embed_queries(["kestrel dashboard"])[0],
                         other.embed_queries(["kestrel dashboard"])[0])

    def test_vectors_are_l2_normalized(self):
        for text in ["", "the the the", "Halberd motion planner", "12345"]:
            vec = self.ef.embed_queries([text])[0]
            norm = sum(v * v for v in vec) ** 0.5
            self.assertAlmostEqual(norm, 1.0, places=6, msg=f"text={text!r}")

    def test_empty_text_does_not_divide_by_zero(self):
        self.assertEqual(len(self.ef.embed_queries([""])[0]), self.ef.dim)

    def test_overlap_beats_no_overlap(self):
        def sq_l2(a, b):
            return sum((x - y) ** 2 for x, y in zip(a, b))

        query = self.ef.embed_queries(["who maintains the Kestrel dashboard"])[0]
        near = self.ef(["Maintains the Kestrel dashboard frontend"])[0]
        far = self.ef(["Raised a Series B round of 42 million dollars"])[0]
        self.assertLess(sq_l2(query, near), sq_l2(query, far))

    def test_documents_and_queries_share_one_space(self):
        text = "Stores fleet telemetry history in TimescaleDB"
        self.assertEqual(self.ef([text])[0], self.ef.embed_queries([text])[0])


class TestScoreQuery(unittest.TestCase):
    """Metric math, checked against hand-computed values."""

    @staticmethod
    def _payload(contents, above_threshold=None):
        results = [{"content": c, "entity_name": "E", "confidence": "HIGH",
                    "relevance_pct": 90.0} for c in contents]
        return {"results": results,
                "above_threshold": len(results) if above_threshold is None
                else above_threshold}

    def test_perfect_hit_at_rank_one(self):
        row = ev.score_query({"id": "t", "query": "q", "expect_contents": ["a"]},
                             self._payload(["a", "b", "c"]), k=5)
        self.assertEqual(row["recall"], 1.0)
        self.assertEqual(row["rr"], 1.0)
        self.assertEqual(row["first_rank"], 1)
        self.assertTrue(row["passed"])

    def test_reciprocal_rank_at_rank_three(self):
        row = ev.score_query({"id": "t", "query": "q", "expect_contents": ["c"]},
                             self._payload(["a", "b", "c"]), k=5)
        self.assertAlmostEqual(row["rr"], 1 / 3)
        self.assertEqual(row["recall"], 1.0)

    def test_miss_scores_zero(self):
        row = ev.score_query({"id": "t", "query": "q", "expect_contents": ["z"]},
                             self._payload(["a", "b", "c"]), k=5)
        self.assertEqual(row["recall"], 0.0)
        self.assertEqual(row["rr"], 0.0)
        self.assertEqual(row["first_rank"], 0)
        self.assertFalse(row["passed"])

    def test_partial_recall_on_multi_answer_query(self):
        row = ev.score_query(
            {"id": "t", "query": "q", "expect_contents": ["a", "z"]},
            self._payload(["a", "b", "c"]), k=5)
        self.assertEqual(row["recall"], 0.5)
        self.assertFalse(row["passed"], "partial recall is not a pass")

    def test_k_cutoff_is_enforced(self):
        row = ev.score_query({"id": "t", "query": "q", "expect_contents": ["e"]},
                             self._payload(["a", "b", "c", "d", "e"]), k=4)
        self.assertEqual(row["recall"], 0.0)

    def test_entity_level_expectation(self):
        payload = {"results": [{"content": "anything", "entity_name": "Kestrel"}],
                   "above_threshold": 1}
        row = ev.score_query({"id": "t", "query": "q", "expect_entities": ["Kestrel"]},
                             payload, k=5)
        self.assertEqual(row["recall"], 1.0)
        self.assertTrue(row["passed"])

    def test_duplicate_hits_do_not_inflate_recall(self):
        row = ev.score_query({"id": "t", "query": "q", "expect_contents": ["a"]},
                             self._payload(["a", "a", "a"]), k=5)
        self.assertEqual(row["recall"], 1.0)

    def test_min3_fallback_is_reported(self):
        row = ev.score_query({"id": "t", "query": "q", "expect_contents": ["a"]},
                             self._payload(["a", "b", "c"], above_threshold=1), k=5)
        self.assertTrue(row["min3_fallback"])
        self.assertEqual(row["above_threshold"], 1)

    def test_empty_expectation_rejected(self):
        with self.assertRaises(ValueError):
            ev.score_query({"id": "t", "query": "q"}, self._payload(["a"]), k=5)


class TestAggregate(unittest.TestCase):
    def test_means(self):
        rows = [
            {"recall": 1.0, "rr": 1.0, "passed": True, "returned": 5,
             "above_threshold": 5, "min3_fallback": False},
            {"recall": 0.0, "rr": 0.0, "passed": False, "returned": 3,
             "above_threshold": 1, "min3_fallback": True},
        ]
        agg = ev.aggregate(rows, k=5)
        self.assertEqual(agg["recall@5"], 0.5)
        self.assertEqual(agg["mrr"], 0.5)
        self.assertEqual(agg["passed"], 1)
        self.assertEqual(agg["failed"], 1)
        self.assertEqual(agg["min3_fallback_queries"], 1)

    def test_empty_rows_do_not_divide_by_zero(self):
        agg = ev.aggregate([], k=5)
        self.assertEqual(agg["recall@5"], 0.0)
        self.assertEqual(agg["queries"], 0)


class TestFixtureModeSmoke(unittest.TestCase):
    """End-to-end: build the vault, run every golden query, check the numbers.

    This exercises the real code path — real store writes, a real ChromaDB
    collection in a temp dir, and the real search_memory ranking / threshold /
    min-3 / JSON-format logic. Only the embedder is substituted.
    """

    @classmethod
    def setUpClass(cls):
        cls.report = ev.run_eval()

    def test_corpus_was_ingested(self):
        self.assertEqual(self.report["ingested_entities"], 10)
        self.assertEqual(self.report["ingested_observations"], 40)
        self.assertEqual(self.report["mode"], "fixture")

    def test_recall_at_5_above_floor(self):
        recall = self.report["aggregate"]["recall@5"]
        self.assertGreaterEqual(
            recall, MIN_RECALL_AT_5,
            f"fixture recall@5 {recall:.3f} < {MIN_RECALL_AT_5} — the ranking "
            f"plumbing regressed (this is not a model-quality signal)",
        )

    def test_mrr_above_floor(self):
        self.assertGreaterEqual(self.report["aggregate"]["mrr"], MIN_MRR)

    def test_every_query_was_scored(self):
        golden = ev.load_golden(ev.DEFAULT_GOLDEN)
        self.assertEqual(self.report["aggregate"]["queries"], len(golden["queries"]))
        self.assertEqual([r["id"] for r in self.report["rows"]],
                         [q["id"] for q in golden["queries"]])

    def test_search_never_returned_empty(self):
        """The min-3 rule means a non-empty vault always yields >= 3 results."""
        for row in self.report["rows"]:
            self.assertGreaterEqual(row["returned"], 3, row["id"])
            self.assertLessEqual(row["returned"], 5, row["id"])

    def test_threshold_gate_is_actually_exercised(self):
        """If everything cleared the bar, the gate would be untested."""
        agg = self.report["aggregate"]
        self.assertGreater(agg["min3_fallback_queries"], 0)

    def test_default_bands_used_when_vault_is_uncalibrated(self):
        self.assertEqual(self.report["thresholds"],
                         {"HIGH": 0.6, "MEDIUM": 1.0, "LOW": 1.4})
        self.assertFalse(self.report["calibrated"])

    def test_run_is_deterministic(self):
        again = ev.run_eval()
        self.assertEqual(
            [(r["id"], r["recall"], r["rr"], r["first_rank"]) for r in again["rows"]],
            [(r["id"], r["recall"], r["rr"], r["first_rank"]) for r in self.report["rows"]],
        )

    def test_temp_data_dir_is_cleaned_up(self):
        self.assertFalse(Path(self.report["data_dir"]).exists())


class TestCalibratedFixtureMode(unittest.TestCase):
    """--calibrate derives bands from the vault's own distances."""

    @classmethod
    def setUpClass(cls):
        cls.report = ev.run_eval(recalibrate=True)

    def test_recall_holds_after_recalibration(self):
        self.assertGreaterEqual(self.report["aggregate"]["recall@5"], MIN_RECALL_AT_5)

    def test_bands_moved_off_the_defaults(self):
        self.assertTrue(self.report["calibrated"])
        self.assertNotEqual(self.report["thresholds"],
                            {"HIGH": 0.6, "MEDIUM": 1.0, "LOW": 1.4})

    def test_bands_are_ordered(self):
        thresholds = self.report["thresholds"]
        self.assertLess(thresholds["HIGH"], thresholds["MEDIUM"])
        self.assertLess(thresholds["MEDIUM"], thresholds["LOW"])

    def test_wider_bands_admit_more_results(self):
        """Sanity check that calibration is doing something observable."""
        self.assertGreater(self.report["aggregate"]["mean_above_threshold"], 3.0)


class TestNoGlobalStateLeaks(unittest.TestCase):
    """The harness repoints module globals; it must put them all back."""

    def test_paths_and_vaults_restored(self):
        import src.config as config
        import src.indexer.embedder as embedder
        import src.indexer.store as store

        before = (config.DATA_DIR, config.CHROMA_DIR, config.ENTITIES_FILE,
                  config.GRAPH_FILE, config.VAULTS_FILE, embedder.CHROMA_DIR,
                  store.DATA_DIR, store.ENTITIES_FILE,
                  store._run_post_write_hooks)
        vaults_before = dict(config.VAULTS)

        ev.run_eval()

        after = (config.DATA_DIR, config.CHROMA_DIR, config.ENTITIES_FILE,
                 config.GRAPH_FILE, config.VAULTS_FILE, embedder.CHROMA_DIR,
                 store.DATA_DIR, store.ENTITIES_FILE,
                 store._run_post_write_hooks)
        self.assertEqual(before, after)
        self.assertEqual(vaults_before, dict(config.VAULTS))
        self.assertNotIn("eval_fixture", config.VAULTS)

    def test_embedder_singleton_restored(self):
        import src.indexer.embedder as embedder

        before = (embedder._embedding_fn, embedder._active_backend,
                  embedder._client)
        ev.run_eval()
        self.assertEqual((embedder._embedding_fn, embedder._active_backend,
                          embedder._client), before)

    def test_real_data_dir_untouched(self):
        """A fixture run must not write into the repo's data/ directory."""
        import src.config as config

        data_dir = PROJECT_ROOT / "data"
        before = sorted(p.name for p in data_dir.iterdir()) if data_dir.exists() else []
        ev.run_eval()
        after = sorted(p.name for p in data_dir.iterdir()) if data_dir.exists() else []
        self.assertEqual(before, after)
        self.assertFalse((data_dir / "eval_fixture_calibration.json").exists())


class TestCliSurface(unittest.TestCase):
    def test_no_ingest_requires_data_dir(self):
        with self.assertRaises(SystemExit):
            ev.main(["--no-ingest"])

    def test_min_recall_gate_passes_on_a_healthy_run(self):
        self.assertEqual(ev.main(["--json", "--min-recall", str(MIN_RECALL_AT_5)]), 0)

    def test_min_recall_gate_fails_when_unreachable(self):
        self.assertEqual(ev.main(["--json", "--min-recall", "1.01"]), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
