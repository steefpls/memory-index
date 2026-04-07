"""Tests for MCP tool implementations."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _make_test_patches(tmpdir):
    """Create standard patches for test isolation."""
    return [
        patch("src.config.DATA_DIR", Path(tmpdir)),
        patch("src.config.ENTITIES_FILE", Path(tmpdir) / "memory_entities.json"),
        patch("src.config.VAULTS_FILE", Path(tmpdir) / "vaults.json"),
        patch("src.config.GRAPH_FILE", Path(tmpdir) / "memory_graph.json"),
        patch("src.config.CHROMA_DIR", Path(tmpdir) / "chroma"),
        patch("src.indexer.store.DATA_DIR", Path(tmpdir)),
        patch("src.indexer.store.ENTITIES_FILE", Path(tmpdir) / "memory_entities.json"),
        patch("src.graph.manager.GRAPH_FILE", Path(tmpdir) / "memory_graph.json"),
        patch("src.graph.manager.DATA_DIR", Path(tmpdir)),
    ]


def _reset_state():
    """Reset all in-memory state."""
    import src.indexer.store as store_mod
    store_mod._entities = {}
    store_mod._observations = {}
    store_mod._loaded = True

    import src.graph.manager as gm
    gm._graph = None
    gm._relations = {}

    import src.config as config_mod
    config_mod.VAULTS = {}


class TestEntityTools(unittest.TestCase):
    """Test entity/observation tool functions."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.patches = _make_test_patches(self.tmpdir)

        # Mock ChromaDB and embedder
        mock_collection = MagicMock()
        mock_ef = MagicMock()
        mock_ef.__call__ = MagicMock(return_value=[[0.1] * 768])
        self.patches.append(patch("src.indexer.store.get_collection", return_value=mock_collection))
        self.patches.append(patch("src.indexer.store.get_embedding_function", return_value=mock_ef))

        for p in self.patches:
            p.start()
        _reset_state()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_create_entity_tool(self):
        from src.tools.entities import tool_create_entity
        result = tool_create_entity("Python", "technology", "test",
                                    observations="General purpose|Created by Guido")
        self.assertIn("Entity created", result)
        self.assertIn("Python", result)
        self.assertIn("technology", result)

    def test_create_entity_empty_name(self):
        from src.tools.entities import tool_create_entity
        result = tool_create_entity("", "technology", "test")
        self.assertIn("Error", result)

    def test_get_entity_tool(self):
        from src.tools.entities import tool_create_entity, tool_get_entity
        tool_create_entity("Python", "technology", "test")
        result = tool_get_entity("Python", "test")
        self.assertIn("Python", result)
        self.assertIn("technology", result)

    def test_get_entity_not_found(self):
        from src.tools.entities import tool_get_entity
        result = tool_get_entity("NonExistent")
        self.assertIn("not found", result)

    def test_list_entities_tool(self):
        from src.tools.entities import tool_create_entity, tool_list_entities
        tool_create_entity("Python", "technology", "test")
        tool_create_entity("Alice", "person", "test")
        result = tool_list_entities(vault="test")
        self.assertIn("Python", result)
        self.assertIn("Alice", result)

    def test_add_observation_tool(self):
        from src.tools.entities import tool_create_entity, tool_add_observation
        tool_create_entity("Python", "technology", "test")
        result = tool_add_observation("Python", "Used in ML", vault="test")
        self.assertIn("Observation added", result)
        self.assertIn("Used in ML", result)

    def test_delete_entity_tool(self):
        from src.tools.entities import tool_create_entity, tool_delete_entity
        tool_create_entity("Python", "technology", "test")
        result = tool_delete_entity("Python", vault="test")
        self.assertIn("Deleted", result)


class TestRelationTools(unittest.TestCase):
    """Test relation tool functions."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.patches = _make_test_patches(self.tmpdir)

        mock_collection = MagicMock()
        mock_ef = MagicMock()
        mock_ef.__call__ = MagicMock(return_value=[[0.1] * 768])
        self.patches.append(patch("src.indexer.store.get_collection", return_value=mock_collection))
        self.patches.append(patch("src.indexer.store.get_embedding_function", return_value=mock_ef))

        for p in self.patches:
            p.start()
        _reset_state()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_create_relation_tool(self):
        from src.tools.entities import tool_create_entity
        from src.tools.relations import tool_create_relation

        tool_create_entity("memory-index", "project", "test")
        tool_create_entity("Python", "technology", "test")

        result = tool_create_relation("memory-index", "Python", "uses", vault="test")
        self.assertIn("Relation created", result)
        self.assertIn("uses", result)

    def test_create_relation_missing_entity(self):
        from src.tools.relations import tool_create_relation
        result = tool_create_relation("nonexistent", "also_nonexistent", "uses")
        self.assertIn("not found", result)

    def test_delete_relation_tool(self):
        from src.tools.entities import tool_create_entity
        from src.tools.relations import tool_create_relation, tool_delete_relation
        from src.graph.manager import get_all_relations

        tool_create_entity("A", "concept", "test")
        tool_create_entity("B", "concept", "test")
        tool_create_relation("A", "B", "related_to", vault="test")

        rels = get_all_relations()
        self.assertEqual(len(rels), 1)
        result = tool_delete_relation(rels[0].id)
        self.assertIn("deleted", result)


class TestStatusTools(unittest.TestCase):
    """Test status and vault tool functions."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.patches = _make_test_patches(self.tmpdir)
        self.patches.append(patch("src.indexer.embedder.get_active_backend", return_value="ONNX + CPU"))

        for p in self.patches:
            p.start()
        _reset_state()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_memory_status(self):
        from src.tools.status import tool_memory_status
        result = tool_memory_status()
        self.assertIn("Memory Index Status", result)
        self.assertIn("Backend", result)

    def test_list_vaults_empty(self):
        from src.tools.status import tool_list_vaults
        result = tool_list_vaults()
        self.assertIn("No vaults", result)

    def test_create_vault(self):
        from src.tools.status import tool_create_vault, tool_list_vaults
        result = tool_create_vault("test")
        self.assertIn("Vault created", result)
        self.assertIn("test", result)

        result2 = tool_list_vaults()
        self.assertIn("test", result2)

    def test_create_vault_empty_name(self):
        from src.tools.status import tool_create_vault
        result = tool_create_vault("")
        self.assertIn("Error", result)

    def test_graph_summary(self):
        from src.tools.status import tool_get_graph_summary
        result = tool_get_graph_summary()
        self.assertIn("Knowledge Graph Summary", result)
        self.assertIn("Nodes: 0", result)

    def test_delete_vault_removes_calibration_file(self):
        from src.tools.status import tool_create_vault, tool_delete_vault

        # Need to mock the chroma client used by tool_delete_vault
        with patch("src.indexer.embedder.get_chroma_client") as mock_client:
            mock_client.return_value = MagicMock()

            tool_create_vault("doomed")

            # Simulate a calibration sidecar being written
            cal_path = Path(self.tmpdir) / "doomed_calibration.json"
            cal_path.write_text("{}", encoding="utf-8")
            self.assertTrue(cal_path.exists())

            tool_delete_vault("doomed")
            self.assertFalse(cal_path.exists())

    def test_delete_vault_no_calibration_file(self):
        """delete_vault should succeed even if no calibration sidecar exists."""
        from src.tools.status import tool_create_vault, tool_delete_vault

        with patch("src.indexer.embedder.get_chroma_client") as mock_client:
            mock_client.return_value = MagicMock()

            tool_create_vault("doomed")
            result = tool_delete_vault("doomed")
            self.assertIn("deleted", result)

    def test_delete_vault_leaves_no_orphans(self):
        """delete_vault should hard-remove its entities, observations, and
        relations — not leave them as orphans for vacuum to collect later."""
        from src.tools.status import tool_create_vault, tool_delete_vault
        from src.tools.entities import tool_create_entity
        from src.tools.relations import tool_create_relation
        import src.indexer.store as store_mod
        import src.graph.manager as gm

        with patch("src.indexer.embedder.get_chroma_client") as mock_client, \
             patch("src.indexer.store.get_collection", return_value=MagicMock()), \
             patch("src.indexer.store.get_embedding_function",
                   return_value=MagicMock(return_value=[[0.1] * 768])):
            mock_client.return_value = MagicMock()

            tool_create_vault("doomed")
            tool_create_entity("A", "concept", "doomed",
                               observations="fact 1|fact 2")
            tool_create_entity("B", "concept", "doomed",
                               observations="fact 3")
            tool_create_relation("A", "B", "related_to", "doomed")

            # Sanity: data exists
            self.assertEqual(
                sum(1 for e in store_mod._entities.values() if e.vault == "doomed"),
                2,
            )
            self.assertGreater(len(store_mod._observations), 0)
            self.assertGreater(len(gm.get_all_relations()), 0)

            tool_delete_vault("doomed")

            # No orphans left in any store
            self.assertEqual(
                sum(1 for e in store_mod._entities.values() if e.vault == "doomed"),
                0,
            )
            self.assertEqual(len(store_mod._observations), 0)
            self.assertEqual(len(gm.get_all_relations()), 0)


if __name__ == "__main__":
    unittest.main()
