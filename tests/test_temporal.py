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
              superseded_by="", superseded_at=None, occurred_at=None):
    obs = Observation(
        id=oid, entity_id=entity_id, content=content,
        source=source, created_at=created_at, superseded_by=superseded_by,
        superseded_at=superseded_at, occurred_at=occurred_at,
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


class TestOccurredAt(unittest.TestCase):
    """Temporal tools order/window on event time when it is known."""

    def setUp(self):
        self.entities = {
            "e1": _make_entity("e1", "Python", "technology"),
            "e2": _make_entity("e2", "Alice", "person"),
        }
        # All three were RECORDED on the same late day, but happened years
        # apart. Only occurred_at can tell them apart.
        self.observations = {
            "o1": _make_obs("o1", "e1", "Created in 1991",
                            "2026-08-01T10:00:00+00:00",
                            occurred_at="1991-02-20"),
            "o2": _make_obs("o2", "e1", "Version 3.0 released",
                            "2026-08-01T10:00:01+00:00",
                            occurred_at="2008-12-03"),
            "o3": _make_obs("o3", "e2", "Joined the team",
                            "2026-08-01T10:00:02+00:00"),
        }

    def test_effective_at_helper(self):
        from src.tools.temporal import _obs_effective_at
        self.assertEqual(_obs_effective_at(self.observations["o1"]), "1991-02-20")
        self.assertEqual(_obs_effective_at(self.observations["o3"]),
                         "2026-08-01T10:00:02+00:00")

    @patch("src.tools.temporal._load_store")
    @patch("src.tools.temporal._observations", new_callable=dict)
    @patch("src.tools.temporal._entities", new_callable=dict)
    def test_timeline_orders_by_occurred_at(self, mock_entities, mock_obs, mock_load):
        mock_entities.update(self.entities)
        mock_obs.update(self.observations)

        from src.tools.temporal import tool_query_timeline
        result = tool_query_timeline(vault="test")

        # Event order (1991, 2008, 2026) — NOT ingestion order, which is
        # o1, o2, o3 by microseconds but would put them all on 2026-08-01.
        self.assertLess(result.find("Created in 1991"),
                        result.find("Version 3.0 released"))
        self.assertLess(result.find("Version 3.0 released"),
                        result.find("Joined the team"))
        # Date grouping uses the event date, not the ingestion date.
        self.assertIn("[1991-02-20]", result)
        self.assertIn("[2008-12-03]", result)

    @patch("src.tools.temporal._load_store")
    @patch("src.tools.temporal._observations", new_callable=dict)
    @patch("src.tools.temporal._entities", new_callable=dict)
    def test_timeline_window_uses_occurred_at(self, mock_entities, mock_obs, mock_load):
        mock_entities.update(self.entities)
        mock_obs.update(self.observations)

        from src.tools.temporal import tool_query_timeline
        result = tool_query_timeline(vault="test", start="1990-01-01",
                                     end="2000-01-01")

        self.assertIn("Created in 1991", result)
        self.assertNotIn("Version 3.0 released", result)
        self.assertNotIn("Joined the team", result)

    @patch("src.tools.temporal._load_store")
    @patch("src.tools.temporal._observations", new_callable=dict)
    @patch("src.tools.temporal._entities", new_callable=dict)
    def test_timeline_falls_back_to_created_at(self, mock_entities, mock_obs, mock_load):
        """An observation with no occurred_at is windowed on created_at."""
        mock_entities.update(self.entities)
        mock_obs.update(self.observations)

        from src.tools.temporal import tool_query_timeline
        result = tool_query_timeline(vault="test", start="2026-01-01")

        self.assertIn("Joined the team", result)
        self.assertNotIn("Created in 1991", result)

    @patch("src.tools.temporal._load_store")
    @patch("src.tools.temporal._observations", new_callable=dict)
    @patch("src.tools.temporal._entities", new_callable=dict)
    def test_timeline_json_exposes_both_times(self, mock_entities, mock_obs, mock_load):
        mock_entities.update(self.entities)
        mock_obs.update(self.observations)

        from src.tools.temporal import tool_query_timeline
        data = json.loads(tool_query_timeline(vault="test", output_format="json"))
        by_content = {i["content"]: i for i in data["timeline"]}

        first = by_content["Created in 1991"]
        self.assertEqual(first["occurred_at"], "1991-02-20")
        self.assertEqual(first["created_at"], "2026-08-01T10:00:00+00:00")
        self.assertEqual(first["effective_at"], "1991-02-20")

        plain = by_content["Joined the team"]
        self.assertIsNone(plain["occurred_at"])
        self.assertEqual(plain["effective_at"], plain["created_at"])

    @patch("src.tools.temporal.resolve_entity")
    @patch("src.tools.temporal._load_store")
    @patch("src.tools.temporal._observations", new_callable=dict)
    @patch("src.tools.temporal._entities", new_callable=dict)
    def test_point_in_time_is_an_as_of_knowledge_snapshot(self, mock_entities,
                                                          mock_obs, mock_load,
                                                          mock_resolve):
        """point_in_time answers "what did we KNOW then", so existence is
        judged on created_at. A fact about 1991 first recorded in 2026 was not
        knowledge in 1995 and must not appear in a 1995 snapshot."""
        mock_entities.update(self.entities)
        mock_obs.update(self.observations)
        mock_resolve.return_value = self.entities["e1"]

        from src.tools.temporal import tool_point_in_time
        result = tool_point_in_time("Python", "1995-01-01")

        self.assertIn("No observations existed", result)

    @patch("src.tools.temporal.resolve_entity")
    @patch("src.tools.temporal._load_store")
    @patch("src.tools.temporal._observations", new_callable=dict)
    @patch("src.tools.temporal._entities", new_callable=dict)
    def test_point_in_time_shows_occurred_at_facts_once_recorded(
        self, mock_entities, mock_obs, mock_load, mock_resolve,
    ):
        """Once recorded, an occurred_at fact is part of the snapshot — and it
        is ordered by event time, not ingestion time."""
        mock_entities.update(self.entities)
        mock_obs.update(self.observations)
        mock_resolve.return_value = self.entities["e1"]

        from src.tools.temporal import tool_point_in_time
        result = tool_point_in_time("Python", "2026-08-02")

        self.assertLess(result.find("Created in 1991"),
                        result.find("Version 3.0 released"))

    @patch("src.tools.temporal.resolve_entity")
    @patch("src.tools.temporal._load_store")
    @patch("src.tools.temporal._observations", new_callable=dict)
    @patch("src.tools.temporal._entities", new_callable=dict)
    def test_point_in_time_supersession_uses_created_at(
        self, mock_entities, mock_obs, mock_load, mock_resolve,
    ):
        """A replacement written AFTER as_of cannot retire its predecessor,
        even when the replacement's occurred_at is far in the past."""
        from src.tools.temporal import tool_point_in_time

        mock_entities.update(self.entities)
        mock_obs.update({
            "v1": _make_obs("v1", "e1", "Runs on .NET Framework",
                            "2026-01-05T00:00:00+00:00", superseded_by="v2"),
            "v2": _make_obs("v2", "e1", "Runs on .NET 8",
                            "2026-02-05T00:00:00+00:00",
                            occurred_at="2019-01-01"),
        })
        mock_resolve.return_value = self.entities["e1"]

        result = tool_point_in_time("Python", "2026-01-15")

        # As of 2026-01-15 only v1 had been written down.
        self.assertIn("Runs on .NET Framework", result)
        self.assertNotIn("Runs on .NET 8", result)

    @patch("src.tools.temporal.resolve_entity")
    @patch("src.tools.temporal._load_store")
    @patch("src.tools.temporal._observations", new_callable=dict)
    @patch("src.tools.temporal._entities", new_callable=dict)
    def test_point_in_time_json_exposes_both_times(self, mock_entities, mock_obs,
                                                   mock_load, mock_resolve):
        mock_entities.update(self.entities)
        mock_obs.update(self.observations)
        mock_resolve.return_value = self.entities["e1"]

        from src.tools.temporal import tool_point_in_time
        data = json.loads(tool_point_in_time("Python", "2026-12-31",
                                             output_format="json"))
        by_content = {o["content"]: o for o in data["observations"]}
        self.assertEqual(by_content["Created in 1991"]["occurred_at"], "1991-02-20")
        self.assertEqual(by_content["Created in 1991"]["effective_at"], "1991-02-20")

    @patch("src.tools.temporal.get_neighbors")
    @patch("src.tools.temporal.get_observations")
    @patch("src.tools.temporal.get_entity")
    @patch("src.tools.temporal.resolve_entity")
    @patch("src.tools.temporal._load_store")
    def test_temporal_neighbors_anchor_on_occurred_at(
        self, mock_load, mock_resolve, mock_get_entity, mock_get_obs, mock_neighbors,
    ):
        """The anchor and neighbor positions come from event time, so a
        neighbor recorded LATER but which happened EARLIER counts as 'before'."""
        target = self.entities["e1"]
        neighbor = self.entities["e2"]
        mock_resolve.return_value = target
        mock_get_entity.return_value = neighbor
        mock_neighbors.return_value = [{
            "entity_id": "e2", "relation_type": "knows",
            "direction": "outgoing", "depth": 1,
        }]

        # Target happened in 2008; neighbor happened in 1999 but was recorded
        # first. Ingestion time alone would call the neighbor 'after'.
        target_obs = [_make_obs("t1", "e1", "Target fact",
                                "2026-08-01T00:00:00+00:00",
                                occurred_at="2008-12-03")]
        neighbor_obs = [_make_obs("n1", "e2", "Neighbor fact",
                                  "2026-08-02T00:00:00+00:00",
                                  occurred_at="1999-05-05")]

        def obs_for(eid, *a, **kw):
            return target_obs if eid == "e1" else neighbor_obs
        mock_get_obs.side_effect = obs_for

        from src.tools.temporal import tool_get_temporal_neighbors
        data = json.loads(tool_get_temporal_neighbors(
            "Python", direction="before", output_format="json"))

        self.assertEqual(data["anchor_time"][:10], "2008-12-03")
        self.assertEqual(len(data["neighbors"]), 1)
        self.assertEqual(data["neighbors"][0]["earliest_observation"][:10],
                         "1999-05-05")


class TestTemporalReadsAreSnapshotted(unittest.TestCase):
    """query_timeline / point_in_time scan a snapshot taken under STORE_LOCK.

    Iterating the live store dicts raced concurrent writes (the auto-librarian
    thread, concurrent HTTP requests) with "dictionary changed size during
    iteration". Probed deterministically by injecting a write mid-scan.
    """

    def setUp(self):
        self.entities = {"e1": _make_entity("e1", "Python", "technology")}
        self.observations = {
            f"o{i}": _make_obs(f"o{i}", "e1", f"fact {i}",
                               f"2026-03-{10 + i:02d}T00:00:00+00:00")
            for i in range(10)
        }

    def _injecting(self, target_dict, real_fn):
        fired = []

        def probe(obs):
            if not fired:
                fired.append(True)
                for i in range(5):
                    target_dict[f"injected{i}"] = _make_obs(
                        f"injected{i}", "e1", f"injected {i}",
                        "2026-04-01T00:00:00+00:00")
            return real_fn(obs)

        return probe, fired

    @patch("src.tools.temporal._load_store")
    @patch("src.tools.temporal._observations", new_callable=dict)
    @patch("src.tools.temporal._entities", new_callable=dict)
    def test_timeline_survives_a_write_landing_mid_scan(self, mock_entities,
                                                        mock_obs, mock_load):
        import src.tools.temporal as temporal

        mock_entities.update(self.entities)
        mock_obs.update(self.observations)

        probe, fired = self._injecting(mock_obs, temporal._obs_effective_dt)
        with patch.object(temporal, "_obs_effective_dt", probe):
            result = temporal.tool_query_timeline(vault="test")  # must not raise

        self.assertTrue(fired)
        self.assertIn("Timeline", result)

    @patch("src.tools.temporal._observations", new_callable=dict)
    @patch("src.tools.temporal._entities", new_callable=dict)
    def test_snapshot_store_returns_copies_taken_under_the_lock(
        self, mock_entities, mock_obs,
    ):
        """Every full-dict scan in this module goes through _snapshot_store."""
        import threading

        import src.indexer.store as store_mod
        import src.tools.temporal as temporal

        mock_entities.update(self.entities)
        mock_obs.update(self.observations)

        self.assertIs(temporal.STORE_LOCK, store_mod.STORE_LOCK)

        # Holding STORE_LOCK from another thread must block the snapshot.
        result = {}
        finished = threading.Event()
        holder_ready = threading.Event()
        release = threading.Event()

        def holder():
            with store_mod.STORE_LOCK:
                holder_ready.set()
                release.wait(5)

        def snapshotter():
            result["value"] = temporal._snapshot_store()
            finished.set()

        h = threading.Thread(target=holder)
        h.start()
        holder_ready.wait(5)

        s = threading.Thread(target=snapshotter)
        s.start()
        self.assertFalse(finished.wait(0.3),
                         "snapshot was taken without holding STORE_LOCK")
        release.set()
        self.assertTrue(finished.wait(5))
        h.join(5)
        s.join(5)

        ents, obs = result["value"]

        # Copies, not aliases: later writes cannot mutate a scan in flight.
        self.assertIsNot(ents, mock_entities)
        self.assertIsNot(obs, mock_obs)
        mock_obs["late"] = _make_obs("late", "e1", "late fact",
                                     "2026-05-01T00:00:00+00:00")
        self.assertNotIn("late", obs)


class TestSupersededAtSemantics(unittest.TestCase):
    """superseded_at turns the supersession pointer into a validity interval."""

    def setUp(self):
        self.entities = {"e1": _make_entity("e1", "Steve", "person")}
        # v1 recorded in January; replacement v2 recorded in June, but the
        # supersession itself was applied late (a gardener pass in July).
        self.observations = {
            "v1": _make_obs("v1", "e1", "Salary is 5k",
                            "2026-01-01T00:00:00+00:00",
                            superseded_by="v2",
                            superseded_at="2026-07-01T04:00:00+00:00"),
            "v2": _make_obs("v2", "e1", "Salary is 5.8k",
                            "2026-06-01T00:00:00+00:00"),
        }

    @patch("src.tools.temporal.resolve_entity")
    @patch("src.tools.temporal._load_store")
    @patch("src.tools.temporal._observations", new_callable=dict)
    @patch("src.tools.temporal._entities", new_callable=dict)
    def test_point_in_time_uses_superseded_at(self, mock_entities, mock_obs,
                                              mock_load, mock_resolve):
        mock_entities.update(self.entities)
        mock_obs.update(self.observations)
        mock_resolve.return_value = self.entities["e1"]

        from src.tools.temporal import tool_point_in_time
        # Mid-June: v2 was already recorded, but the supersession of v1 was
        # only applied in July — the store still believed v1 at this point.
        result = tool_point_in_time("e1", "2026-06-15")
        self.assertIn("Salary is 5k", result)
        self.assertIn("Salary is 5.8k", result)

        # After the supersession was applied, v1 drops out.
        result = tool_point_in_time("e1", "2026-07-02")
        self.assertNotIn("Salary is 5k\n", result + "\n")
        self.assertIn("Salary is 5.8k", result)

    @patch("src.tools.temporal.resolve_entity")
    @patch("src.tools.temporal._load_store")
    @patch("src.tools.temporal._observations", new_callable=dict)
    @patch("src.tools.temporal._entities", new_callable=dict)
    def test_point_in_time_legacy_fallback(self, mock_entities, mock_obs,
                                           mock_load, mock_resolve):
        """Rows superseded before the field existed fall back to the
        replacement's created_at, exactly as before."""
        mock_entities.update(self.entities)
        self.observations["v1"].superseded_at = None
        mock_obs.update(self.observations)
        mock_resolve.return_value = self.entities["e1"]

        from src.tools.temporal import tool_point_in_time
        # Without superseded_at, v2's created_at (June 1) is the cutover.
        result = tool_point_in_time("e1", "2026-06-15")
        self.assertNotIn("Salary is 5k\n", result + "\n")
        self.assertIn("Salary is 5.8k", result)

    @patch("src.tools.temporal._load_store")
    @patch("src.tools.temporal._observations", new_callable=dict)
    @patch("src.tools.temporal._entities", new_callable=dict)
    def test_timeline_hides_superseded_by_default(self, mock_entities, mock_obs, mock_load):
        mock_entities.update(self.entities)
        mock_obs.update(self.observations)

        from src.tools.temporal import tool_query_timeline
        result = tool_query_timeline(vault="test")
        self.assertNotIn("Salary is 5k\n", result + "\n")
        self.assertIn("Salary is 5.8k", result)

    @patch("src.tools.temporal._load_store")
    @patch("src.tools.temporal._observations", new_callable=dict)
    @patch("src.tools.temporal._entities", new_callable=dict)
    def test_timeline_include_superseded_labels_the_old_fact(self, mock_entities,
                                                            mock_obs, mock_load):
        mock_entities.update(self.entities)
        mock_obs.update(self.observations)

        from src.tools.temporal import tool_query_timeline
        result = tool_query_timeline(vault="test", include_superseded=True)
        self.assertIn("Salary is 5k", result)
        self.assertIn("[superseded 2026-07-01]", result)

    @patch("src.tools.temporal._load_store")
    @patch("src.tools.temporal._observations", new_callable=dict)
    @patch("src.tools.temporal._entities", new_callable=dict)
    def test_timeline_date_axis_record_vs_event(self, mock_entities, mock_obs,
                                                mock_load):
        """The same window gives axis-dependent answers, by explicit choice."""
        mock_entities.update(self.entities)
        mock_obs.update({
            # Recorded in 2026, about an event in 1991.
            "h1": _make_obs("h1", "e1", "Born in Singapore",
                            "2026-06-01T00:00:00+00:00",
                            occurred_at="1991-02-20"),
        })

        from src.tools.temporal import tool_query_timeline
        event_view = tool_query_timeline(vault="test", start="1991-01-01",
                                         end="1992-01-01")
        self.assertIn("Born in Singapore", event_view)

        record_view = tool_query_timeline(vault="test", start="1991-01-01",
                                          end="1992-01-01", date_axis="record")
        self.assertNotIn("Born in Singapore", record_view)

        record_2026 = tool_query_timeline(vault="test", start="2026-01-01",
                                          end="2027-01-01", date_axis="record")
        self.assertIn("Born in Singapore", record_2026)

    def test_invalid_date_axis_rejected(self):
        from src.tools.temporal import tool_query_timeline
        self.assertIn("date_axis", tool_query_timeline(date_axis="wat"))


if __name__ == "__main__":
    unittest.main()
