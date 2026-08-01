"""Tests for maintenance tools (vacuum) and delete_entity relation cleanup."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestMaintenance(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.patches = []
        tmp = Path(self.tmpdir)

        path_patches = [
            ("src.config.DATA_DIR", tmp),
            ("src.config.ENTITIES_FILE", tmp / "memory_entities.json"),
            ("src.config.VAULTS_FILE", tmp / "vaults.json"),
            ("src.config.GRAPH_FILE", tmp / "memory_graph.json"),
            ("src.config.CHROMA_DIR", tmp / "chroma"),
            ("src.indexer.store.DATA_DIR", tmp),
            ("src.indexer.store.ENTITIES_FILE", tmp / "memory_entities.json"),
            ("src.graph.manager.GRAPH_FILE", tmp / "memory_graph.json"),
            ("src.graph.manager.DATA_DIR", tmp),
        ]
        for target, value in path_patches:
            p = patch(target, value)
            self.patches.append(p)
            p.start()

        import src.indexer.store as store_mod
        store_mod._entities = {}
        store_mod._observations = {}
        store_mod._loaded = True

        import src.graph.manager as gm
        gm._graph = None
        gm._relations = {}

        import src.config as config_mod
        config_mod.VAULTS = {
            "alpha": config_mod.VaultConfig(name="alpha", collection_name="memory_alpha"),
        }

        self.mock_collection = MagicMock()
        p = patch("src.indexer.store.get_collection", return_value=self.mock_collection)
        self.patches.append(p)
        p.start()

        self.mock_ef = MagicMock(return_value=[[0.1] * 768])
        p = patch("src.indexer.store.get_embedding_function", return_value=self.mock_ef)
        self.patches.append(p)
        p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    # ---- delete_entity now cleans relations ----

    def test_delete_entity_removes_its_relations(self):
        from src.indexer.store import create_entity, delete_entity
        from src.graph.manager import add_relation, get_all_relations
        from src.models.relation import Relation

        a = create_entity("A", "concept", "alpha")
        b = create_entity("B", "concept", "alpha")
        c = create_entity("C", "concept", "alpha")
        add_relation(Relation(id="r1", from_entity=a.id, to_entity=b.id, relation_type="uses"))
        add_relation(Relation(id="r2", from_entity=b.id, to_entity=c.id, relation_type="uses"))
        add_relation(Relation(id="r3", from_entity=a.id, to_entity=c.id, relation_type="uses"))

        delete_entity(b.id)

        remaining = get_all_relations()
        # r1 and r2 both involve B and should be gone; r3 (a->c) remains
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0].from_entity, a.id)
        self.assertEqual(remaining[0].to_entity, c.id)

    # ---- vacuum ----

    def test_vacuum_removes_orphan_vault_entities(self):
        from src.indexer.store import create_entity
        from src.tools.maintenance import tool_vacuum_store
        import src.indexer.store as store_mod

        # Add an entity in alpha and one in a vault that no longer exists
        a = create_entity("Alive", "concept", "alpha")

        # Manually inject an entity in a non-existent vault (simulates leftover after delete_vault)
        from src.models.entity import Entity
        ghost = Entity(id="ghost1", name="Ghost", entity_type="concept", vault="ghost_vault")
        store_mod._entities[ghost.id] = ghost

        result = tool_vacuum_store()
        self.assertIn("1 orphan vault", result)
        self.assertNotIn(ghost.id, store_mod._entities)
        self.assertIn(a.id, store_mod._entities)

    def test_vacuum_removes_soft_deleted_entities(self):
        from src.indexer.store import create_entity, delete_entity
        from src.tools.maintenance import tool_vacuum_store
        import src.indexer.store as store_mod

        a = create_entity("A", "concept", "alpha")
        b = create_entity("B", "concept", "alpha")
        delete_entity(a.id)

        # Sanity: A is soft-deleted but still in store
        self.assertIn(a.id, store_mod._entities)
        self.assertTrue(store_mod._entities[a.id].deleted)

        result = tool_vacuum_store()
        self.assertIn("1 soft-deleted", result)
        self.assertNotIn(a.id, store_mod._entities)
        self.assertIn(b.id, store_mod._entities)

    def test_vacuum_removes_dangling_relations(self):
        from src.indexer.store import create_entity
        from src.graph.manager import add_relation, get_all_relations
        from src.models.relation import Relation
        from src.tools.maintenance import tool_vacuum_store

        a = create_entity("A", "concept", "alpha")
        # Manually add a relation pointing to a non-existent entity
        add_relation(Relation(
            id="dangling", from_entity="missing_id", to_entity=a.id,
            relation_type="related_to",
        ))
        self.assertEqual(len(get_all_relations()), 1)

        result = tool_vacuum_store()
        self.assertIn("Relations to remove: 1", result)
        self.assertEqual(len(get_all_relations()), 0)

    def test_vacuum_cascades_observations(self):
        from src.indexer.store import create_entity, delete_entity
        from src.tools.maintenance import tool_vacuum_store
        import src.indexer.store as store_mod

        a = create_entity("A", "concept", "alpha",
                          observations=["fact 1", "fact 2"])
        delete_entity(a.id)

        # Soft-deleted obs still in store before vacuum
        self.assertEqual(len(store_mod._observations), 2)

        tool_vacuum_store()
        self.assertEqual(len(store_mod._observations), 0)

    def test_vacuum_dry_run_does_not_modify(self):
        from src.indexer.store import create_entity, delete_entity
        from src.tools.maintenance import tool_vacuum_store
        import src.indexer.store as store_mod

        a = create_entity("A", "concept", "alpha")
        delete_entity(a.id)
        self.assertIn(a.id, store_mod._entities)

        result = tool_vacuum_store(dry_run=True)
        self.assertIn("dry run", result)
        self.assertIn(a.id, store_mod._entities)  # untouched

    def test_vacuum_idempotent_on_clean_store(self):
        from src.indexer.store import create_entity
        from src.tools.maintenance import tool_vacuum_store

        create_entity("A", "concept", "alpha")
        result = tool_vacuum_store()
        self.assertIn("Entities to remove: 0", result)
        self.assertIn("Observations to remove: 0", result)
        self.assertIn("Relations to remove: 0", result)

    # ---- concurrency ----

    def test_vacuum_scan_and_removal_are_one_critical_section(self):
        """A concurrent write must not be able to land between vacuum's scan
        and its hard-pop + _save_store.

        Probed deterministically from inside the scan (via VAULTS lookup)
        rather than by racing threads: with the scan unlocked the writer runs
        straight through; holding STORE_LOCK across scan and removal blocks it
        until the vacuum is done.
        """
        import threading
        from src.indexer.store import create_entity, delete_entity, add_observation
        import src.indexer.store as store_mod
        import src.tools.maintenance as maint

        keep = create_entity("Keep", "concept", "alpha")
        junk = create_entity("Junk", "concept", "alpha",
                             observations=["junk fact"])
        delete_entity(junk.id)

        probe = {}
        writer_started = threading.Event()
        writer_done = threading.Event()

        def writer():
            writer_started.set()
            add_observation(keep.id, "concurrent fact")
            writer_done.set()

        class ProbingVaults(dict):
            """Fires once, at the point in the scan that reads a vault."""

            def __getitem__(self, key):
                if "fired" not in probe:
                    probe["fired"] = True
                    t = threading.Thread(target=writer)
                    probe["thread"] = t
                    t.start()
                    writer_started.wait(5)
                    probe["ran_during_scan"] = writer_done.wait(0.5)
                return super().__getitem__(key)

        with patch.object(maint, "VAULTS", ProbingVaults(maint.VAULTS)):
            result = maint.tool_vacuum_store()

        probe["thread"].join(5)

        self.assertTrue(probe.get("fired"), "probe never ran — scan changed shape")
        self.assertFalse(
            probe["ran_during_scan"],
            "a write completed while vacuum was mid-scan: the scan and the "
            "hard-pop are not one critical section",
        )

        # The vacuum still did its job, and the concurrent write survived.
        self.assertIn("1 soft-deleted", result)
        self.assertNotIn(junk.id, store_mod._entities)
        self.assertIn(keep.id, store_mod._entities)
        contents = {o.content for o in store_mod._observations.values()}
        self.assertIn("concurrent fact", contents)


if __name__ == "__main__":
    unittest.main()
