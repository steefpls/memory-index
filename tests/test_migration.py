"""Tests for the one-time legacy-JSON -> SQLite migration and the DAL."""

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


LEGACY_VAULTS = {
    "vaults": {
        "work": {"name": "work", "collection_name": "memory_work",
                 "created_at": "2026-03-13T00:00:00+00:00"},
    }
}

LEGACY_ENTITIES = {
    "entities": [
        {"id": "e1", "name": "Steve", "entity_type": "person", "vault": "work",
         "created_at": "2026-03-13T00:00:00+00:00",
         "updated_at": "2026-03-14T00:00:00+00:00", "deleted": False},
        {"id": "e2", "name": "OldThing", "entity_type": "project", "vault": "work",
         "created_at": "2026-03-13T00:00:00+00:00",
         "updated_at": "2026-03-13T00:00:00+00:00", "deleted": True},
    ],
    "observations": [
        {"id": "o1", "entity_id": "e1", "content": "Salary is 5k",
         "source": "chat", "created_at": "2026-01-01T00:00:00+00:00",
         "deleted": False, "superseded_by": "o2"},
        {"id": "o2", "entity_id": "e1", "content": "Salary is 5.8k",
         "source": "chat", "created_at": "2026-06-01T00:00:00+00:00",
         "deleted": False, "occurred_at": "2026-05-28"},
    ],
}

LEGACY_GRAPH = {
    "relations": [
        {"id": "r1", "from_entity": "e1", "to_entity": "e2",
         "relation_type": "created", "weight": 1.0, "context": "",
         "created_at": "2026-03-13T00:00:00+00:00"},
    ]
}


class TestLegacyMigration(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        tmp = Path(self.tmpdir)
        (tmp / "vaults.json").write_text(json.dumps(LEGACY_VAULTS), encoding="utf-8")
        (tmp / "memory_entities.json").write_text(json.dumps(LEGACY_ENTITIES), encoding="utf-8")
        (tmp / "memory_graph.json").write_text(json.dumps(LEGACY_GRAPH), encoding="utf-8")

        self.patches = [
            patch("src.config.DB_FILE", tmp / "memory.db"),
            patch("src.config.VAULTS_FILE", tmp / "vaults.json"),
            patch("src.config.ENTITIES_FILE", tmp / "memory_entities.json"),
            patch("src.config.GRAPH_FILE", tmp / "memory_graph.json"),
        ]
        for p in self.patches:
            p.start()

        import src.indexer.db as db_mod
        self.db = db_mod
        self.db.reset()

    def tearDown(self):
        self.db.reset()
        for p in self.patches:
            p.stop()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_fresh_db_imports_all_legacy_rows(self):
        ents, obs = self.db.load_entities_observations()
        self.assertEqual({e["id"] for e in ents}, {"e1", "e2"})
        self.assertEqual({o["id"] for o in obs}, {"o1", "o2"})

        rels = self.db.load_relations()
        self.assertEqual([r["id"] for r in rels], ["r1"])

        vaults = self.db.get_all_vaults()
        self.assertEqual([v["name"] for v in vaults], ["work"])

    def test_migration_preserves_flags_and_pointers(self):
        ents, obs = self.db.load_entities_observations()
        by_id = {o["id"]: o for o in obs}
        self.assertEqual(by_id["o1"]["superseded_by"], "o2")
        self.assertIsNone(by_id["o1"]["superseded_at"])  # legacy: no stamp
        self.assertEqual(by_id["o2"]["occurred_at"], "2026-05-28")

        ent_by_id = {e["id"]: e for e in ents}
        self.assertTrue(ent_by_id["e2"]["deleted"])
        self.assertFalse(ent_by_id["e1"]["deleted"])

    def test_json_files_left_untouched(self):
        self.db.load_entities_observations()  # trigger migration
        raw = json.loads(
            (Path(self.tmpdir) / "memory_entities.json").read_text(encoding="utf-8"))
        self.assertEqual(raw, LEGACY_ENTITIES)

    def test_migration_runs_once(self):
        """A reopened DB must not re-import (or duplicate) legacy rows."""
        ents, _ = self.db.load_entities_observations()
        self.assertEqual(len(ents), 2)

        # Delete a row, then force a fresh connection: the DB file already
        # exists, so the legacy JSON must NOT be re-imported.
        self.db.hard_delete_entities(["e2"])
        self.db.reset()
        ents, _ = self.db.load_entities_observations()
        self.assertEqual({e["id"] for e in ents}, {"e1"})

    def test_store_loads_through_migration(self):
        """The store's normal load path reads the migrated rows."""
        import src.indexer.store as store_mod
        with patch.object(store_mod, "_entities", {}), \
             patch.object(store_mod, "_observations", {}), \
             patch.object(store_mod, "_loaded", False):
            store_mod._load_store()
            ent = store_mod.get_entity("e1")
            self.assertIsNotNone(ent)
            self.assertEqual(ent.name, "Steve")
            obs = store_mod.get_observations("e1", include_superseded=True)
            self.assertEqual(len(obs), 2)


class TestDalRoundTrip(unittest.TestCase):
    """Row-level upserts and hard deletes round-trip through SQLite."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.patches = [
            patch("src.config.DB_FILE", Path(self.tmpdir) / "memory.db"),
        ]
        for p in self.patches:
            p.start()
        import src.indexer.db as db_mod
        self.db = db_mod
        self.db.reset()

    def tearDown(self):
        self.db.reset()
        for p in self.patches:
            p.stop()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_observation_superseded_at_round_trips(self):
        from src.models.observation import Observation
        obs = Observation(id="o1", entity_id="e1", content="x",
                          superseded_by="o2",
                          superseded_at="2026-07-01T00:00:00+00:00")
        self.db.upsert_observations([obs])
        _, rows = self.db.load_entities_observations()
        self.assertEqual(rows[0]["superseded_at"], "2026-07-01T00:00:00+00:00")

        restored = Observation.from_dict(rows[0])
        self.assertEqual(restored.superseded_at, "2026-07-01T00:00:00+00:00")
        self.assertTrue(restored.is_superseded)

    def test_upsert_overwrites_by_id(self):
        from src.models.entity import Entity
        self.db.upsert_entities([Entity(id="e1", name="Old", entity_type="person",
                                        vault="work")])
        self.db.upsert_entities([Entity(id="e1", name="New", entity_type="person",
                                        vault="work")])
        ents, _ = self.db.load_entities_observations()
        self.assertEqual(len(ents), 1)
        self.assertEqual(ents[0]["name"], "New")

    def test_mark_superseded_never_fabricates_a_timestamp(self):
        """An unstamped supersession must stay unstamped.

        mark_superseded is a reconstruction primitive (import / chain rebuild).
        Defaulting an empty superseded_at to now() collapsed every restored
        legacy supersession onto the restore date, which silently rewrote
        point_in_time history for the whole vault.
        """
        from src.models.entity import Entity
        from src.models.observation import Observation
        import src.indexer.store as store_mod

        with patch.object(store_mod, "_entities", {}), \
             patch.object(store_mod, "_observations", {}), \
             patch.object(store_mod, "_loaded", True):
            ent = Entity(id="e1", name="A", entity_type="person", vault="work")
            old = Observation(id="o1", entity_id="e1", content="v1")
            new = Observation(id="o2", entity_id="e1", content="v2")
            store_mod._entities["e1"] = ent
            store_mod._observations["o1"] = old
            store_mod._observations["o2"] = new

            self.assertTrue(store_mod.mark_superseded("o1", "o2"))
            self.assertTrue(old.is_superseded)
            self.assertIsNone(
                old.superseded_at,
                "empty superseded_at must stay None so point_in_time falls "
                "back to the replacement's created_at",
            )

            # An explicitly supplied stamp is still honoured verbatim.
            self.assertTrue(
                store_mod.mark_superseded("o1", "o2",
                                          superseded_at="2026-05-02T07:26:07+00:00"))
            self.assertEqual(old.superseded_at, "2026-05-02T07:26:07+00:00")

    def test_hard_deletes(self):
        from src.models.entity import Entity
        from src.models.observation import Observation
        self.db.upsert_entities([Entity(id="e1", name="A", entity_type="person",
                                        vault="work")])
        self.db.upsert_observations([Observation(id="o1", entity_id="e1",
                                                 content="x")])
        self.db.hard_delete_observations(["o1"])
        self.db.hard_delete_entities(["e1"])
        ents, obs = self.db.load_entities_observations()
        self.assertEqual((ents, obs), ([], []))


if __name__ == "__main__":
    unittest.main()
