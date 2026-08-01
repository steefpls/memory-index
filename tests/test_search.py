"""Tests for observation-level search: flat ranking, threshold gate, min-3,
semantic default, and honestly-scored graph expansion."""

import json
import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Calibrated bands used by every search test below.
TEST_THRESHOLDS = {"HIGH": 0.6, "MEDIUM": 1.0, "LOW": 1.4}


def _row(oid, eid, dist, content, name=None, etype="technology",
         vault="test", source="", created_at="2026-01-01", superseded_by=None):
    meta = {
        "entity_id": eid,
        "entity_name": name or eid.upper(),
        "entity_type": etype,
        "content": content,
        "source": source,
        "vault": vault,
        "created_at": created_at,
    }
    if superseded_by:
        meta["superseded_by"] = superseded_by
    return {"id": oid, "dist": dist, "meta": meta}


def _cond_matches(cond, meta):
    """Minimal Chroma `where` evaluator — enough for the filters search uses.

    Deliberately mirrors real ChromaDB's *validation*, not just its matching:
    the range operators reject non-numeric operands exactly as chromadb does.
    An earlier version of this fake happily compared ISO strings with `$gte`,
    which made a completely broken since/before filter look like it worked.
    """
    if cond is None:
        return True
    if "$and" in cond:
        return all(_cond_matches(c, meta) for c in cond["$and"])
    for field, expr in cond.items():
        val = meta.get(field)
        if isinstance(expr, dict):
            for op, operand in expr.items():
                if op == "$in":
                    if val not in operand:
                        return False
                elif op in ("$gte", "$gt", "$lte", "$lt"):
                    if isinstance(operand, bool) or not isinstance(operand, (int, float)):
                        raise ValueError(
                            f"Expected operand value to be an int or a float "
                            f"for operator {op}, got {operand} in query."
                        )
                    if val is None or not isinstance(val, (int, float)):
                        return False
                    if op == "$gte" and val < operand:
                        return False
                    if op == "$gt" and val <= operand:
                        return False
                    if op == "$lte" and val > operand:
                        return False
                    if op == "$lt" and val >= operand:
                        return False
                else:
                    raise AssertionError(f"unsupported operator {op}")
        elif val != expr:
            return False
    return True


def _restricts_entity_id(cond) -> bool:
    if not cond:
        return False
    if "$and" in cond:
        return any(_restricts_entity_id(c) for c in cond["$and"])
    return "entity_id" in cond


class FakeCollection:
    """Stand-in for a Chroma collection that honours where/n_results.

    `hidden_entities` emulates observations that fall outside the semantic
    top-k window: an open query never surfaces them, but a query explicitly
    restricted to their entity_id does. That is exactly the situation the
    associative strategy exists for.
    """

    def __init__(self, rows, hidden_entities=()):
        self.rows = rows
        self.hidden_entities = set(hidden_entities)
        self.queries = []

    def query(self, query_embeddings=None, n_results=10, where=None,
              include=None):
        self.queries.append({"n_results": n_results, "where": where})
        matched = [r for r in self.rows if _cond_matches(where, r["meta"])]
        if not _restricts_entity_id(where):
            matched = [r for r in matched
                       if r["meta"]["entity_id"] not in self.hidden_entities]
        matched = sorted(matched, key=lambda r: r["dist"])[:n_results]
        return {
            "ids": [[r["id"] for r in matched]],
            "metadatas": [[r["meta"] for r in matched]],
            "distances": [[r["dist"] for r in matched]],
            "documents": [[r["meta"]["content"] for r in matched]],
        }


class SearchTestCase(unittest.TestCase):
    """Base fixture: one 'test' vault backed by a FakeCollection."""

    rows: list = []
    hidden_entities: tuple = ()

    def setUp(self):
        from src.tools.search import _calibration_cache
        from src.models.observation import Observation
        from src.models.entity import Entity

        self.collections = {
            "test": FakeCollection(list(self.rows), self.hidden_entities),
        }
        _calibration_cache["test"] = dict(TEST_THRESHOLDS)
        _calibration_cache["other"] = dict(TEST_THRESHOLDS)

        # Search joins Chroma hits back to the store, so the fixture rows must
        # exist as store objects too — the meta dict now only feeds the fake
        # collection's `where` evaluation.
        self.store_observations: dict = {}
        self.store_entities: dict = {}
        for r in self.rows:
            meta = r["meta"]
            self.store_observations[r["id"]] = Observation(
                id=r["id"],
                entity_id=meta["entity_id"],
                content=meta["content"],
                source=meta.get("source", ""),
                created_at=meta.get("created_at", ""),
                superseded_by=meta.get("superseded_by", ""),
            )
            self.store_entities[meta["entity_id"]] = Entity(
                id=meta["entity_id"],
                name=meta.get("entity_name", meta["entity_id"].upper()),
                entity_type=meta.get("entity_type", "technology"),
                vault=meta.get("vault", "test"),
            )

        self.spread = MagicMock(return_value={})
        self.get_entity = MagicMock(
            side_effect=lambda eid: self.store_entities.get(eid))

        self.patches = [
            patch("src.tools.search.VAULTS", {"test": object()}),
            patch("src.tools.search.get_vault",
                  side_effect=lambda v: types.SimpleNamespace(collection_name=v)),
            patch("src.tools.search.get_collection",
                  side_effect=lambda name: self.collections[name]),
            patch("src.tools.search._get_query_embeddings_with_guard",
                  return_value=[[0.1] * 8]),
            patch("src.tools.search.spread_activation", self.spread),
            patch("src.tools.search.get_entity", self.get_entity),
            patch("src.tools.search.get_observation",
                  side_effect=lambda oid: self.store_observations.get(oid)),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        from src.tools.search import _calibration_cache
        for p in self.patches:
            p.stop()
        _calibration_cache.clear()

    def search_json(self, **kwargs):
        from src.tools.search import search_memory
        kwargs.setdefault("output_format", "json")
        return json.loads(search_memory("a query", **kwargs))


class TestFlatObservationRanking(SearchTestCase):
    """The retrieval unit is the observation, not the entity."""

    rows = [
        _row("o1", "e1", 0.10, "python is a language"),
        _row("o2", "e1", 0.20, "python has a GIL"),
        _row("o3", "e1", 0.30, "python ships with asyncio"),
        _row("o4", "e2", 0.40, "rust has no GC"),
        _row("o5", "e2", 0.50, "rust has traits"),
        _row("o6", "e3", 0.55, "go has goroutines"),
        _row("o7", "e3", 0.58, "go has channels"),
        _row("o8", "e4", 0.59, "zig has comptime"),
    ]

    def test_returns_multiple_observations_per_entity(self):
        payload = self.search_json(vault="test")
        ids = [r["observation_id"] for r in payload["results"]]
        # Old behaviour collapsed to one row per entity; now e1 contributes
        # three separately-ranked observations.
        self.assertEqual(ids[:3], ["o1", "o2", "o3"])
        self.assertEqual([r["entity_id"] for r in payload["results"][:3]],
                         ["e1", "e1", "e1"])

    def test_default_n_results_is_five(self):
        payload = self.search_json(vault="test")
        self.assertEqual(len(payload["results"]), 5)
        self.assertEqual(payload["returned"], 5)

    def test_ranked_best_first(self):
        payload = self.search_json(vault="test", n_results=8)
        dists = [r["distance"] for r in payload["results"]]
        self.assertEqual(dists, sorted(dists))
        pcts = [r["relevance_pct"] for r in payload["results"]]
        self.assertEqual(pcts, sorted(pcts, reverse=True))

    def test_json_is_flat_observation_list(self):
        payload = self.search_json(vault="test")
        first = payload["results"][0]
        for key in ("rank", "observation_id", "content", "source", "entity_id",
                    "entity_name", "entity_type", "vault", "distance",
                    "relevance_pct", "confidence", "above_threshold",
                    "graph_boosted", "superseded"):
            self.assertIn(key, first)
        # No entity-nested observation bundles anymore
        self.assertNotIn("observations", first)

    def test_n_results_honoured_above_min(self):
        payload = self.search_json(vault="test", n_results=7)
        self.assertEqual(len(payload["results"]), 7)

    def test_text_groups_consecutive_same_entity(self):
        from src.tools.search import search_memory
        text = search_memory("a query", vault="test", n_results=5)
        # One header per entity run, but a numbered line per observation.
        self.assertEqual(text.count("id=e1"), 1)
        self.assertEqual(text.count("id=e2"), 1)
        for marker in ("[1]", "[2]", "[3]", "[4]", "[5]"):
            self.assertIn(marker, text)
        self.assertIn("python has a GIL", text)


class TestThresholdGate(SearchTestCase):
    """Only observations above the calibrated noise floor are returned."""

    rows = [
        _row("o1", "e1", 0.10, "clear hit one"),
        _row("o2", "e1", 0.20, "clear hit two"),
        _row("o3", "e2", 0.30, "clear hit three"),
        _row("o4", "e2", 1.50, "noise one"),      # > LOW (1.4) => NO MATCH
        _row("o5", "e3", 1.80, "noise two"),
        _row("o6", "e3", 2.00, "noise three"),
    ]

    def test_below_threshold_excluded(self):
        payload = self.search_json(vault="test", n_results=5)
        self.assertEqual([r["observation_id"] for r in payload["results"]],
                         ["o1", "o2", "o3"])
        self.assertTrue(all(r["above_threshold"] for r in payload["results"]))
        self.assertEqual(payload["above_threshold"], 3)


class TestMinThreeRule(SearchTestCase):
    """Fewer than 3 above threshold => best 3 overall, honest labels."""

    rows = [
        _row("o1", "e1", 0.10, "the only real hit"),
        _row("o2", "e2", 1.50, "weak one"),
        _row("o3", "e2", 1.60, "weak two"),
        _row("o4", "e3", 1.70, "weak three"),
    ]

    def test_returns_three_when_one_clears(self):
        payload = self.search_json(vault="test", n_results=5)
        self.assertEqual(len(payload["results"]), 3)
        self.assertEqual(payload["above_threshold"], 1)
        self.assertEqual([r["observation_id"] for r in payload["results"]],
                         ["o1", "o2", "o3"])

    def test_fallback_results_keep_honest_labels(self):
        payload = self.search_json(vault="test", n_results=5)
        self.assertEqual(payload["results"][0]["confidence"], "HIGH")
        for r in payload["results"][1:]:
            self.assertFalse(r["above_threshold"])
            self.assertEqual(r["confidence"], "NO MATCH")

    def test_text_notes_the_shortfall(self):
        from src.tools.search import search_memory
        text = search_memory("a query", vault="test")
        self.assertIn("threshold", text.splitlines()[0])
        self.assertIn("1 result(s)", text.splitlines()[0])


class TestMinThreeWithTinyVault(SearchTestCase):
    """Vault holds fewer than 3 matches — return what exists, not an error."""

    rows = [
        _row("o1", "e1", 1.50, "weak one"),
        _row("o2", "e1", 1.60, "weak two"),
    ]

    def test_returns_all_available(self):
        payload = self.search_json(vault="test", n_results=5)
        self.assertEqual(len(payload["results"]), 2)
        self.assertEqual(payload["above_threshold"], 0)


class TestFilters(SearchTestCase):
    rows = [
        _row("o1", "e1", 0.10, "recent tech fact", created_at="2026-05-01"),
        _row("o2", "e1", 0.15, "old tech fact", created_at="2025-01-01"),
        _row("o3", "e2", 0.20, "person fact", etype="person",
             created_at="2026-05-01"),
        _row("o4", "e1", 0.05, "replaced fact", superseded_by="o1"),
    ]

    def test_superseded_hidden_by_default(self):
        payload = self.search_json(vault="test", n_results=5)
        ids = [r["observation_id"] for r in payload["results"]]
        self.assertNotIn("o4", ids)

    def test_include_superseded(self):
        payload = self.search_json(vault="test", n_results=5,
                                   include_superseded=True)
        results = {r["observation_id"]: r for r in payload["results"]}
        self.assertIn("o4", results)
        self.assertTrue(results["o4"]["superseded"])

    def test_entity_type_filter(self):
        payload = self.search_json(vault="test", n_results=5,
                                   entity_type="person")
        self.assertEqual([r["observation_id"] for r in payload["results"]],
                         ["o3"])

    def test_since_filter(self):
        payload = self.search_json(vault="test", n_results=5, since="2026-01-01")
        ids = [r["observation_id"] for r in payload["results"]]
        self.assertNotIn("o2", ids)
        self.assertIn("o1", ids)

    def test_before_filter(self):
        payload = self.search_json(vault="test", n_results=5, before="2026-01-01")
        ids = [r["observation_id"] for r in payload["results"]]
        self.assertEqual(ids, ["o2"])

    def test_since_and_before_combined(self):
        payload = self.search_json(vault="test", n_results=5,
                                   since="2026-01-02", before="2026-06-01")
        ids = [r["observation_id"] for r in payload["results"]]
        self.assertEqual(set(ids), {"o1", "o3"})

    def test_date_filters_never_reach_the_where_clause(self):
        """Chroma rejects string operands for $gte/$lt, so date windows must
        be applied in Python — never pushed into the vector store filter."""
        self.search_json(vault="test", n_results=5,
                         since="2026-01-01", before="2026-12-31")
        for q in self.collections["test"].queries:
            self.assertNotIn("created_at", json.dumps(q["where"] or {}))

    def test_date_filter_combines_with_entity_type(self):
        payload = self.search_json(vault="test", n_results=5,
                                   entity_type="person", since="2026-01-01")
        self.assertEqual([r["observation_id"] for r in payload["results"]],
                         ["o3"])

    def test_invalid_since_is_reported(self):
        from src.tools.search import search_memory
        result = search_memory("a query", vault="test", since="not-a-date")
        self.assertIn("Error", result)
        self.assertIn("since", result)

    def test_invalid_date_axis_is_reported(self):
        from src.tools.search import search_memory
        result = search_memory("a query", vault="test", date_axis="wat")
        self.assertIn("Error", result)
        self.assertIn("date_axis", result)


class TestDateAxis(SearchTestCase):
    """since/before can window on record time (default) or event time."""

    rows = [
        # Recorded May 2026, no event time.
        _row("o1", "e1", 0.10, "plain recent fact", created_at="2026-05-01"),
        # Recorded May 2026 about a 1991 event.
        _row("o2", "e1", 0.20, "historical fact", created_at="2026-05-01"),
        _row("o3", "e1", 0.30, "another recent fact", created_at="2026-05-02"),
    ]

    def setUp(self):
        super().setUp()
        self.store_observations["o2"].occurred_at = "1991-02-20"

    def test_record_axis_is_default(self):
        payload = self.search_json(vault="test", since="2026-01-01")
        ids = {r["observation_id"] for r in payload["results"]}
        self.assertEqual(ids, {"o1", "o2", "o3"})

    def test_event_axis_windows_on_occurred_at(self):
        payload = self.search_json(vault="test", since="2026-01-01",
                                   date_axis="event")
        ids = {r["observation_id"] for r in payload["results"]}
        self.assertEqual(ids, {"o1", "o3"})  # o2 happened in 1991

        payload = self.search_json(vault="test", since="1991-01-01",
                                   before="1992-01-01", date_axis="event")
        ids = {r["observation_id"] for r in payload["results"]}
        self.assertIn("o2", ids)


class TestStrategyDefaults(SearchTestCase):
    rows = [
        _row("o1", "e1", 0.10, "hit one"),
        _row("o2", "e1", 0.20, "hit two"),
        _row("o3", "e2", 0.30, "hit three"),
    ]

    def test_semantic_is_the_default(self):
        self.search_json(vault="test")
        self.spread.assert_not_called()

    def test_unknown_strategy_falls_back_to_semantic(self):
        self.search_json(vault="test", strategy="wat")
        self.spread.assert_not_called()

    def test_associative_is_opt_in(self):
        self.search_json(vault="test", strategy="associative")
        self.spread.assert_called_once()


class TestAssociativeScoring(SearchTestCase):
    """Graph neighbours must earn their place with a real score."""

    rows = [
        _row("o1", "e1", 0.10, "seed hit one"),
        _row("o2", "e1", 0.20, "seed hit two"),
        _row("o3", "e1", 0.30, "seed hit three"),
        # e9 is only reachable via the graph; one of its facts is genuinely
        # relevant, the other is noise.
        _row("n1", "e9", 0.35, "neighbour that really matches",
             name="NEIGHBOUR", etype="concept"),
        _row("n2", "e9", 1.90, "neighbour noise", name="NEIGHBOUR",
             etype="concept"),
    ]
    hidden_entities = ("e9",)

    def setUp(self):
        super().setUp()
        self.spread.return_value = {"e9": 0.7}

    def test_neighbour_scored_with_real_distance(self):
        payload = self.search_json(vault="test", strategy="associative")
        by_id = {r["observation_id"]: r for r in payload["results"]}
        self.assertIn("n1", by_id)
        self.assertEqual(by_id["n1"]["distance"], 0.35)
        self.assertTrue(by_id["n1"]["graph_boosted"])
        self.assertTrue(by_id["n1"]["above_threshold"])

    def test_no_fabricated_distances(self):
        payload = self.search_json(vault="test", strategy="associative",
                                   n_results=10)
        real = {r["dist"] for r in self.rows}
        for r in payload["results"]:
            self.assertIn(r["distance"], real)
        # The old bug pinned neighbours to the worst observed distance.
        worst = max(real)
        self.assertNotIn(worst, [r["distance"] for r in payload["results"]])

    def test_graph_query_is_restricted_to_candidates(self):
        self.search_json(vault="test", strategy="associative")
        graph_queries = [q for q in self.collections["test"].queries
                         if q["where"] is not None]
        self.assertTrue(graph_queries)
        self.assertEqual(graph_queries[-1]["where"], {"entity_id": {"$in": ["e9"]}})

    def test_neighbour_noise_still_gated(self):
        payload = self.search_json(vault="test", strategy="associative",
                                   n_results=10)
        self.assertNotIn("n2", [r["observation_id"] for r in payload["results"]])


class TestAssociativeDegradesGracefully(SearchTestCase):
    """Neighbour clears nothing => associative == semantic."""

    rows = [
        _row("o1", "e1", 0.10, "seed hit one"),
        _row("o2", "e1", 0.20, "seed hit two"),
        _row("o3", "e1", 0.30, "seed hit three"),
        _row("n1", "e9", 1.95, "irrelevant neighbour", name="NEIGHBOUR"),
    ]
    hidden_entities = ("e9",)

    def setUp(self):
        super().setUp()
        self.spread.return_value = {"e9": 0.9}

    def test_matches_semantic_result(self):
        associative = self.search_json(vault="test", strategy="associative")
        semantic = self.search_json(vault="test", strategy="semantic")
        self.assertEqual([r["observation_id"] for r in associative["results"]],
                         [r["observation_id"] for r in semantic["results"]])
        self.assertFalse(any(r["graph_boosted"] for r in associative["results"]))


class TestEmptyAndErrors(SearchTestCase):
    rows = []

    def test_no_results(self):
        from src.tools.search import search_memory
        self.assertIn("No results found", search_memory("a query", vault="test"))

    def test_unknown_vault(self):
        from src.tools.search import search_memory
        self.assertIn("Unknown vault", search_memory("a query", vault="nope"))

    def test_bad_output_format(self):
        from src.tools.search import search_memory
        self.assertIn("Error", search_memory("a query", vault="test",
                                             output_format="yaml"))


class TestSearchScoring(unittest.TestCase):
    """Test confidence labels and normalized scores."""

    def test_confidence_labels(self):
        from src.tools.search import _confidence_label, _calibration_cache

        # Set up known thresholds
        _calibration_cache["test"] = {"HIGH": 650, "MEDIUM": 775, "LOW": 875}

        self.assertEqual(_confidence_label(500, "test"), "HIGH")
        self.assertEqual(_confidence_label(700, "test"), "MEDIUM")
        self.assertEqual(_confidence_label(800, "test"), "LOW")
        self.assertEqual(_confidence_label(900, "test"), "NO MATCH")

        _calibration_cache.clear()

    def test_normalized_score(self):
        from src.tools.search import _normalized_score, _calibration_cache

        _calibration_cache["test"] = {"HIGH": 650, "MEDIUM": 775, "LOW": 875}

        # Perfect match
        self.assertEqual(_normalized_score(0, "test"), 100.0)

        # HIGH boundary -> 85%
        score_at_high = _normalized_score(650, "test")
        self.assertAlmostEqual(score_at_high, 85.0, places=0)

        # MEDIUM boundary -> 55%
        score_at_med = _normalized_score(775, "test")
        self.assertAlmostEqual(score_at_med, 55.0, places=0)

        # LOW boundary -> 15%
        score_at_low = _normalized_score(875, "test")
        self.assertAlmostEqual(score_at_low, 15.0, places=0)

        # Far away -> 0%
        score_beyond = _normalized_score(1200, "test")
        self.assertEqual(score_beyond, 0.0)

        _calibration_cache.clear()

    def test_normalized_score_monotonic(self):
        """Scores should decrease as distance increases."""
        from src.tools.search import _normalized_score, _calibration_cache

        _calibration_cache["test"] = {"HIGH": 650, "MEDIUM": 775, "LOW": 875}

        prev_score = 100.0
        for dist in range(0, 1100, 50):
            score = _normalized_score(dist, "test")
            self.assertLessEqual(score, prev_score + 0.1,
                                 f"Score increased at distance {dist}")
            prev_score = score

        _calibration_cache.clear()

    def test_format_text(self):
        from src.tools.search import _format_text, _calibration_cache

        _calibration_cache["test"] = {"HIGH": 650, "MEDIUM": 775, "LOW": 875}

        results = [
            {
                "observation_id": "o1",
                "entity_id": "e1",
                "entity_name": "Python",
                "entity_type": "technology",
                "vault": "test",
                "distance": 500,
                "content": "General purpose language",
                "source": "docs",
                "graph_boosted": False,
            },
            {
                "observation_id": "o2",
                "entity_id": "e1",
                "entity_name": "Python",
                "entity_type": "technology",
                "vault": "test",
                "distance": 700,
                "content": "Created by Guido",
                "source": "",
                "graph_boosted": False,
            },
        ]

        text = _format_text(results, "test query", above_threshold_count=2)
        self.assertIn("Python", text)
        self.assertIn("technology", text)
        self.assertIn("General purpose language", text)
        self.assertIn("Created by Guido", text)
        self.assertIn("[src: docs]", text)
        # Entity context line is emitted once for the consecutive run
        self.assertEqual(text.count("(technology)"), 1)
        # No shortfall note when everything cleared
        self.assertNotIn("threshold", text)

        _calibration_cache.clear()

    def test_format_json_is_flat(self):
        from src.tools.search import _format_json, _calibration_cache

        _calibration_cache["test"] = {"HIGH": 650, "MEDIUM": 775, "LOW": 875}

        results = [{
            "observation_id": "o1",
            "entity_id": "e1",
            "entity_name": "Python",
            "entity_type": "technology",
            "vault": "test",
            "distance": 500,
            "content": "General purpose language",
            "source": "docs",
            "graph_boosted": False,
        }]
        payload = json.loads(_format_json(results, "q", strategy="semantic",
                                          above_threshold_count=1))
        self.assertEqual(payload["results"][0]["observation_id"], "o1")
        self.assertEqual(payload["results"][0]["rank"], 1)
        self.assertEqual(payload["strategy"], "semantic")

        _calibration_cache.clear()


if __name__ == "__main__":
    unittest.main()
