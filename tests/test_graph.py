"""Tests for graph manager and traversal."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestGraphManager(unittest.TestCase):
    """Test NetworkX graph operations."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.patches = []

        from tests.support import patch_sqlite
        patch_sqlite(self.tmpdir, self.patches)

        # Reset graph state
        import src.graph.manager as gm
        gm._graph = None
        gm._relations = {}

    def tearDown(self):
        from tests.support import close_sqlite
        close_sqlite()
        for p in self.patches:
            p.stop()
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_add_relation(self):
        from src.graph.manager import add_relation, get_relation
        from src.models.relation import Relation

        rel = Relation(
            id="r1",
            from_entity="e1",
            to_entity="e2",
            relation_type="uses",
        )
        add_relation(rel)

        retrieved = get_relation("r1")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.from_entity, "e1")
        self.assertEqual(retrieved.to_entity, "e2")
        self.assertEqual(retrieved.relation_type, "uses")

    def test_remove_relation(self):
        from src.graph.manager import add_relation, remove_relation, get_relation
        from src.models.relation import Relation

        rel = Relation(id="r1", from_entity="e1", to_entity="e2", relation_type="uses")
        add_relation(rel)
        result = remove_relation("r1")
        self.assertTrue(result)
        self.assertIsNone(get_relation("r1"))

    def test_get_relations_for_entity(self):
        from src.graph.manager import add_relation, get_relations_for_entity
        from src.models.relation import Relation

        add_relation(Relation(id="r1", from_entity="e1", to_entity="e2", relation_type="uses"))
        add_relation(Relation(id="r2", from_entity="e2", to_entity="e3", relation_type="depends_on"))
        add_relation(Relation(id="r3", from_entity="e3", to_entity="e1", relation_type="related_to"))

        rels = get_relations_for_entity("e1")
        self.assertEqual(len(rels), 2)  # r1 (outgoing) and r3 (incoming)

    def test_remove_entity_relations(self):
        from src.graph.manager import add_relation, remove_entity_relations, get_relation_count
        from src.models.relation import Relation

        add_relation(Relation(id="r1", from_entity="e1", to_entity="e2", relation_type="uses"))
        add_relation(Relation(id="r2", from_entity="e2", to_entity="e1", relation_type="depends_on"))
        add_relation(Relation(id="r3", from_entity="e3", to_entity="e4", relation_type="related_to"))

        count = remove_entity_relations("e1")
        self.assertEqual(count, 2)
        self.assertEqual(get_relation_count(), 1)

    def test_graph_lock_is_reentrant(self):
        """add_relation -> _save_graph re-enters, so a plain Lock would hang."""
        import threading
        import src.graph.manager as gm

        self.assertIsInstance(gm.GRAPH_LOCK, type(threading.RLock()))
        with gm.GRAPH_LOCK:
            with gm.GRAPH_LOCK:
                pass

    def test_concurrent_writers_all_persist(self):
        """Row-level writes: concurrent add_relations must all land in SQLite."""
        import threading
        import src.graph.manager as gm
        import src.indexer.db as db_mod
        from src.models.relation import Relation

        def writer(i):
            gm.add_relation(Relation(id=f"r{i}", from_entity=f"a{i}",
                                     to_entity=f"b{i}", relation_type="uses"))

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(5)

        self.assertEqual(gm.get_relation_count(), 8)
        self.assertEqual(len(db_mod.load_relations()), 8)

    def test_persistence(self):
        import src.indexer.db as db_mod
        from src.graph.manager import add_relation
        from src.models.relation import Relation

        add_relation(Relation(id="r1", from_entity="e1", to_entity="e2", relation_type="uses"))

        rows = db_mod.load_relations()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["id"], "r1")

    def test_load_from_db(self):
        import src.indexer.db as db_mod
        from src.models.relation import Relation

        # Seed the DB directly, then force a fresh in-memory load
        db_mod.upsert_relations([
            Relation(id="r1", from_entity="e1", to_entity="e2",
                     relation_type="uses"),
        ])

        import src.graph.manager as gm
        gm._graph = None
        gm._relations = {}

        from src.graph.manager import get_relation
        rel = get_relation("r1")
        self.assertIsNotNone(rel)
        self.assertEqual(rel.relation_type, "uses")


class TestGraphTraversal(unittest.TestCase):
    """Test graph traversal utilities."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.patches = []

        from tests.support import patch_sqlite
        patch_sqlite(self.tmpdir, self.patches)

        import src.graph.manager as gm
        gm._graph = None
        gm._relations = {}

        # Build a test graph: e1 -> e2 -> e3, e1 -> e4
        from src.graph.manager import add_relation
        from src.models.relation import Relation

        add_relation(Relation(id="r1", from_entity="e1", to_entity="e2", relation_type="uses"))
        add_relation(Relation(id="r2", from_entity="e2", to_entity="e3", relation_type="depends_on"))
        add_relation(Relation(id="r3", from_entity="e1", to_entity="e4", relation_type="related_to"))

    def tearDown(self):
        from tests.support import close_sqlite
        close_sqlite()
        for p in self.patches:
            p.stop()
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_get_neighbors_depth_1(self):
        from src.graph.traversal import get_neighbors

        neighbors = get_neighbors("e1", max_depth=1)
        neighbor_ids = {n["entity_id"] for n in neighbors}
        self.assertIn("e2", neighbor_ids)
        self.assertIn("e4", neighbor_ids)
        self.assertNotIn("e3", neighbor_ids)

    def test_get_neighbors_depth_2(self):
        from src.graph.traversal import get_neighbors

        neighbors = get_neighbors("e1", max_depth=2)
        neighbor_ids = {n["entity_id"] for n in neighbors}
        self.assertIn("e2", neighbor_ids)
        self.assertIn("e3", neighbor_ids)
        self.assertIn("e4", neighbor_ids)

    def test_get_neighbors_filter_type(self):
        from src.graph.traversal import get_neighbors

        neighbors = get_neighbors("e1", max_depth=1, relation_type="uses")
        self.assertEqual(len(neighbors), 1)
        self.assertEqual(neighbors[0]["entity_id"], "e2")

    def test_get_graph_summary(self):
        from src.graph.traversal import get_graph_summary

        summary = get_graph_summary()
        self.assertEqual(summary["nodes"], 4)
        self.assertEqual(summary["edges"], 3)
        self.assertIn("uses", summary["relation_types"])
        self.assertIn("depends_on", summary["relation_types"])

    def test_spreading_activation(self):
        from src.graph.traversal import spread_activation

        # e1 is seed, should activate e2 and e4 (1-hop neighbors)
        activated = spread_activation({"e1"}, decay=0.7, max_hops=2, top_k=10)
        self.assertIn("e2", activated)
        self.assertIn("e4", activated)
        self.assertNotIn("e1", activated)  # seeds excluded from results
        # e3 is 2 hops away, should be activated at max_hops=2
        self.assertIn("e3", activated)
        # e2 (1 hop) should have more energy than e3 (2 hops)
        self.assertGreater(activated["e2"], activated["e3"])


if __name__ == "__main__":
    unittest.main()
