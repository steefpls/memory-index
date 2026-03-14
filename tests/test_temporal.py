"""Tests for temporal query tools — timeline, point-in-time, temporal neighbors."""

import json
import os
import sys
import unittest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.entity import Entity
from src.models.observation import Observation


def _make_entity(eid, name, etype="concept", vault="test"):
    return Entity(id=eid, name=name, entity_type=etype, vault=vault)


def _make_obs(oid, entity_id, content, created_at, source="",
              superseded_by=""):
    obs = Observation(
        id=oid, entity_id=entity_id, content=content,
        source=source, created_at=created_at, superseded_by=superseded_by,
    )
    return obs


class TestParseIso(unittest.TestCase):
    def test_full_iso(self):
        from src.tools.temporal import _parse_iso
        dt = _parse_iso("2026-03-13T10:00:00+00:00")
        self.assertIsNotNone(dt)
        self.assertEqual(dt.year, 2026)
        self.assertEqual(dt.month, 3)

    def test_date_only(self):
        from src.tools.temporal import _parse_iso
        dt = _parse_iso("2026-03-13")
        self.assertIsNotNone(dt)
        self.assertEqual(dt.day, 13)

    def test_empty(self):
        from src.tools.temporal import _parse_iso
        self.assertIsNone(_parse_iso(""))
        self.assertIsNone(_parse_iso(None))

    def test_invalid(self):
        from src.tools.temporal import _parse_iso
        self.assertIsNone(_parse_iso("not-a-date"))


class TestQueryTimeline(unittest.TestCase):
    def setUp(self):
        """Set up mock entities and observations."""
        self.entities = {
            "e1": _make_entity("e1", "Python", "technology"),
            "e2": _make_entity("e2", "Alice", "person"),
        }
        self.observations = {
            "o1": _make_obs("o1", "e1", "General purpose language",
                           "2026-03-10T08:00:00+00:00"),
            "o2": _make_obs("o2", "e1", "Version 3.12 released",
                           "2026-03-12T10:00:00+00:00"),
            "o3": _make_obs("o3", "e2", "Works on backend",
                           "2026-03-11T14:00:00+00:00"),
            "o4": _make_obs("o4", "e2", "Promoted to lead",
                           "2026-03-13T09:00:00+00:00"),
        }

    @patch("src.tools.temporal._load_store")
    @patch("src.tools.temporal._observations", new_callable=dict)
    @patch("src.tools.temporal._entities", new_callable=dict)
    def test_full_range(self, mock_entities, mock_obs, mock_load):
        mock_entities.update(self.entities)
        mock_obs.update(self.observations)

        from src.tools.temporal import tool_query_timeline
        result = tool_query_timeline(vault="test")

        self.assertIn("Timeline", result)
        self.assertIn("Python", result)
        self.assertIn("Alice", result)
        # Should be chronological
        python_pos = result.find("General purpose language")
        alice_pos = result.find("Works on backend")
        self.assertLess(python_pos, alice_pos)

    @patch("src.tools.temporal._load_store")
    @patch("src.tools.temporal._observations", new_callable=dict)
    @patch("src.tools.temporal._entities", new_callable=dict)
    def test_date_filter(self, mock_entities, mock_obs, mock_load):
        mock_entities.update(self.entities)
        mock_obs.update(self.observations)

        from src.tools.temporal import tool_query_timeline
        result = tool_query_timeline(vault="test", start="2026-03-12",
                                      end="2026-03-13")

        self.assertIn("Version 3.12 released", result)
        self.assertNotIn("General purpose language", result)
        self.assertNotIn("Promoted to lead", result)

    @patch("src.tools.temporal._load_store")
    @patch("src.tools.temporal._observations", new_callable=dict)
    @patch("src.tools.temporal._entities", new_callable=dict)
    def test_entity_type_filter(self, mock_entities, mock_obs, mock_load):
        mock_entities.update(self.entities)
        mock_obs.update(self.observations)

        from src.tools.temporal import tool_query_timeline
        result = tool_query_timeline(vault="test", entity_type="person")

        self.assertIn("Alice", result)
        self.assertNotIn("Python", result)

    @patch("src.tools.temporal._load_store")
    @patch("src.tools.temporal._observations", new_callable=dict)
    @patch("src.tools.temporal._entities", new_callable=dict)
    def test_json_output(self, mock_entities, mock_obs, mock_load):
        mock_entities.update(self.entities)
        mock_obs.update(self.observations)

        from src.tools.temporal import tool_query_timeline
        result = tool_query_timeline(vault="test", output_format="json")

        data = json.loads(result)
        self.assertIn("timeline", data)
        self.assertIn("count", data)
        self.assertEqual(data["count"], 4)

    @patch("src.tools.temporal._load_store")
    @patch("src.tools.temporal._observations", new_callable=dict)
    @patch("src.tools.temporal._entities", new_callable=dict)
    def test_empty_results(self, mock_entities, mock_obs, mock_load):
        mock_entities.update(self.entities)
        mock_obs.update(self.observations)

        from src.tools.temporal import tool_query_timeline
        result = tool_query_timeline(vault="test", start="2030-01-01")
        self.assertIn("No observations found", result)

    def test_invalid_date(self):
        from src.tools.temporal import tool_query_timeline
        result = tool_query_timeline(start="not-a-date")
        self.assertIn("Error", result)

    @patch("src.tools.temporal._load_store")
    @patch("src.tools.temporal._observations", new_callable=dict)
    @patch("src.tools.temporal._entities", new_callable=dict)
    def test_skips_superseded(self, mock_entities, mock_obs, mock_load):
        """Superseded observations should be excluded from timeline."""
        mock_entities.update(self.entities)
        obs = dict(self.observations)
        obs["o1"] = _make_obs("o1", "e1", "Old fact",
                              "2026-03-10T08:00:00+00:00",
                              superseded_by="o2")
        mock_obs.update(obs)

        from src.tools.temporal import tool_query_timeline
        result = tool_query_timeline(vault="test")
        self.assertNotIn("Old fact", result)


class TestPointInTime(unittest.TestCase):
    def setUp(self):
        self.entities = {
            "e1": _make_entity("e1", "Python", "technology"),
        }
        self.observations = {
            "o1": _make_obs("o1", "e1", "Version 3.11",
                           "2026-03-10T08:00:00+00:00",
                           superseded_by="o2"),
            "o2": _make_obs("o2", "e1", "Version 3.12",
                           "2026-03-12T10:00:00+00:00"),
            "o3": _make_obs("o3", "e1", "Used at company X",
                           "2026-03-11T14:00:00+00:00"),
        }

    @patch("src.tools.temporal.resolve_entity")
    @patch("src.tools.temporal._load_store")
    @patch("src.tools.temporal._observations", new_callable=dict)
    @patch("src.tools.temporal._entities", new_callable=dict)
    def test_before_supersede(self, mock_entities, mock_obs, mock_load,
                               mock_resolve):
        """Before o2 was created, o1 should still be visible."""
        mock_entities.update(self.entities)
        mock_obs.update(self.observations)
        mock_resolve.return_value = self.entities["e1"]

        from src.tools.temporal import tool_point_in_time
        result = tool_point_in_time("Python", "2026-03-11T00:00:00+00:00")

        self.assertIn("Version 3.11", result)
        self.assertNotIn("Version 3.12", result)

    @patch("src.tools.temporal.resolve_entity")
    @patch("src.tools.temporal._load_store")
    @patch("src.tools.temporal._observations", new_callable=dict)
    @patch("src.tools.temporal._entities", new_callable=dict)
    def test_after_supersede(self, mock_entities, mock_obs, mock_load,
                              mock_resolve):
        """After o2 was created, o1 should be hidden and o2 shown."""
        mock_entities.update(self.entities)
        mock_obs.update(self.observations)
        mock_resolve.return_value = self.entities["e1"]

        from src.tools.temporal import tool_point_in_time
        result = tool_point_in_time("Python", "2026-03-13T00:00:00+00:00")

        self.assertNotIn("Version 3.11", result)
        self.assertIn("Version 3.12", result)
        self.assertIn("Used at company X", result)

    @patch("src.tools.temporal.resolve_entity")
    @patch("src.tools.temporal._load_store")
    @patch("src.tools.temporal._observations", new_callable=dict)
    @patch("src.tools.temporal._entities", new_callable=dict)
    def test_json_output(self, mock_entities, mock_obs, mock_load,
                          mock_resolve):
        mock_entities.update(self.entities)
        mock_obs.update(self.observations)
        mock_resolve.return_value = self.entities["e1"]

        from src.tools.temporal import tool_point_in_time
        result = tool_point_in_time("Python", "2026-03-13",
                                     output_format="json")

        data = json.loads(result)
        self.assertEqual(data["entity_name"], "Python")
        self.assertEqual(data["as_of"], "2026-03-13")
        self.assertEqual(len(data["observations"]), 2)

    @patch("src.tools.temporal.resolve_entity")
    def test_entity_not_found(self, mock_resolve):
        mock_resolve.return_value = None

        from src.tools.temporal import tool_point_in_time
        result = tool_point_in_time("NonExistent", "2026-03-13")
        self.assertIn("not found", result)

    def test_invalid_date(self):
        from src.tools.temporal import tool_point_in_time
        result = tool_point_in_time("Python", "garbage")
        self.assertIn("Error", result)

    @patch("src.tools.temporal.resolve_entity")
    @patch("src.tools.temporal._load_store")
    @patch("src.tools.temporal._observations", new_callable=dict)
    @patch("src.tools.temporal._entities", new_callable=dict)
    def test_before_any_observations(self, mock_entities, mock_obs,
                                      mock_load, mock_resolve):
        """Querying before any observations exist should return empty."""
        mock_entities.update(self.entities)
        mock_obs.update(self.observations)
        mock_resolve.return_value = self.entities["e1"]

        from src.tools.temporal import tool_point_in_time
        result = tool_point_in_time("Python", "2020-01-01")
        self.assertIn("No observations existed", result)


class TestRRFMerge(unittest.TestCase):
    """Test Reciprocal Rank Fusion scoring."""

    def test_basic_merge(self):
        from src.tools.search import _rrf_merge

        vector = [
            {"entity_id": "a", "distance": 100},
            {"entity_id": "b", "distance": 200},
            {"entity_id": "c", "distance": 300},
        ]
        graph = [
            {"entity_id": "d", "_energy": 0.9},
            {"entity_id": "b", "_energy": 0.5},
        ]

        scores = _rrf_merge(vector, graph)

        # 'a' only in vector (rank 1): 1/(60+1)
        self.assertAlmostEqual(scores["a"], 1/61, places=6)
        # 'b' in both (vector rank 2 + graph rank 2): 1/(60+2) + 1/(60+2)
        self.assertAlmostEqual(scores["b"], 1/62 + 1/62, places=6)
        # 'd' only in graph (rank 1): 1/(60+1)
        self.assertAlmostEqual(scores["d"], 1/61, places=6)

    def test_entity_in_both_ranks_higher(self):
        from src.tools.search import _rrf_merge

        vector = [
            {"entity_id": "a", "distance": 100},
            {"entity_id": "b", "distance": 200},
        ]
        graph = [
            {"entity_id": "b", "_energy": 0.9},
            {"entity_id": "c", "_energy": 0.5},
        ]

        scores = _rrf_merge(vector, graph)

        # 'b' appears in both lists, should have highest composite score
        self.assertGreater(scores["b"], scores["a"])
        self.assertGreater(scores["b"], scores["c"])

    def test_empty_graph(self):
        from src.tools.search import _rrf_merge
        vector = [{"entity_id": "a", "distance": 100}]
        scores = _rrf_merge(vector, [])
        self.assertEqual(len(scores), 1)
        self.assertIn("a", scores)

    def test_custom_k(self):
        from src.tools.search import _rrf_merge
        vector = [{"entity_id": "a", "distance": 100}]
        graph = [{"entity_id": "a", "_energy": 0.5}]

        scores_k10 = _rrf_merge(vector, graph, k=10)
        scores_k60 = _rrf_merge(vector, graph, k=60)

        # Lower k gives higher scores (more spread between ranks)
        self.assertGreater(scores_k10["a"], scores_k60["a"])


if __name__ == "__main__":
    unittest.main()
