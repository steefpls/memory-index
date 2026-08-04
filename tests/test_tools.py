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
        patch("src.config.DB_FILE", Path(tmpdir) / "memory.db"),
        patch("src.config.CHROMA_DIR", Path(tmpdir) / "chroma"),
    ]


def _reset_state():
    """Reset all in-memory state (call AFTER the patches are started)."""
    import src.indexer.db as db_mod
    db_mod.reset()

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
        from tests.support import close_sqlite
        close_sqlite()
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
        self.assertIn("Python", result)
        self.assertIn("id=", result)
        # Response should NOT echo the observation content back to the caller —
        # the caller just sent it. Saves tokens on every write.
        self.assertNotIn("Used in ML", result)

    def test_add_observation_tool_supersedes(self):
        from src.tools.entities import tool_create_entity, tool_add_observation
        tool_create_entity("Python", "technology", "test")
        first = tool_add_observation("Python", "Old fact", vault="test")
        old_id = first.split("id=", 1)[1].split(",", 1)[0].strip()
        result = tool_add_observation(
            "Python", "New fact", vault="test", supersedes=old_id,
        )
        self.assertIn("Observation added", result)
        self.assertIn(f"supersedes={old_id}", result)
        self.assertNotIn("New fact", result)

    def test_delete_entity_tool(self):
        from src.tools.entities import tool_create_entity, tool_delete_entity
        tool_create_entity("Python", "technology", "test")
        result = tool_delete_entity("Python", vault="test")
        self.assertIn("Deleted", result)

    def test_get_entity_json_carries_ids(self):
        """JSON output must expose observation AND relation IDs — the text
        form prints neither reliably, so it can't drive an edit surface."""
        import json as _json
        from src.tools.entities import (tool_create_entity, tool_add_observation,
                                        tool_get_entity)
        from src.tools.relations import tool_create_relation

        tool_create_entity("Python", "technology", "test")
        tool_create_entity("Django", "technology", "test")
        tool_add_observation("Python", "Used in ML", vault="test",
                             source="chat 2026-08-04")
        tool_create_relation("Django", "Python", "uses", vault="test")

        payload = _json.loads(tool_get_entity("Python", "test",
                                              output_format="json"))
        self.assertEqual(payload["entity"]["name"], "Python")
        self.assertEqual(payload["entity"]["type"], "technology")
        self.assertEqual(payload["observations_total"], 1)
        obs = payload["observations"][0]
        self.assertTrue(obs["id"])
        self.assertEqual(obs["content"], "Used in ML")
        self.assertEqual(obs["source"], "chat 2026-08-04")
        self.assertFalse(obs["superseded"])
        rel = payload["relations"][0]
        self.assertTrue(rel["id"])
        self.assertEqual(rel["direction"], "in")
        self.assertEqual(rel["type"], "uses")
        self.assertEqual(rel["other_name"], "Django")

    def test_get_entity_json_separates_superseded(self):
        import json as _json
        from src.tools.entities import (tool_create_entity, tool_add_observation,
                                        tool_get_entity)

        tool_create_entity("Python", "technology", "test")
        first = tool_add_observation("Python", "Old fact", vault="test")
        old_id = first.split("id=", 1)[1].split(",", 1)[0].strip()
        tool_add_observation("Python", "New fact", vault="test",
                             supersedes=old_id)

        payload = _json.loads(tool_get_entity(
            "Python", "test", include_superseded=True, output_format="json"))
        self.assertEqual([o["content"] for o in payload["observations"]],
                         ["New fact"])
        old = payload["superseded"][0]
        self.assertEqual(old["id"], old_id)
        self.assertTrue(old["superseded"])
        self.assertEqual(old["superseded_by"], payload["observations"][0]["id"])
        self.assertTrue(old["superseded_at"])

    def test_get_entity_json_not_found(self):
        import json as _json
        from src.tools.entities import tool_get_entity
        payload = _json.loads(tool_get_entity("NonExistent",
                                              output_format="json"))
        self.assertEqual(payload["error"], "not_found")

    def test_undelete_observation_tool(self):
        from src.tools.entities import (tool_create_entity, tool_add_observation,
                                        tool_delete_observation,
                                        tool_undelete_observation)
        tool_create_entity("Python", "technology", "test")
        added = tool_add_observation("Python", "Used in ML", vault="test")
        obs_id = added.split("id=", 1)[1].split(",", 1)[0].strip()

        self.assertIn("deleted", tool_delete_observation(obs_id))
        self.assertIn("restored", tool_undelete_observation(obs_id).lower())
        # Second undelete is a no-op, not a lie.
        self.assertIn("Cannot undelete", tool_undelete_observation(obs_id))

    def test_delete_observation_tool_reports_revivals(self):
        from src.tools.entities import (tool_create_entity, tool_add_observation,
                                        tool_delete_observation)
        tool_create_entity("Python", "technology", "test")
        first = tool_add_observation("Python", "Old fact", vault="test")
        old_id = first.split("id=", 1)[1].split(",", 1)[0].strip()
        second = tool_add_observation("Python", "New fact", vault="test",
                                      supersedes=old_id)
        new_id = second.split("id=", 1)[1].split(",", 1)[0].strip()

        result = tool_delete_observation(new_id)
        self.assertIn("revived 1 superseded", result)
        self.assertIn(old_id, result)


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
        from tests.support import close_sqlite
        close_sqlite()
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
        from tests.support import close_sqlite
        close_sqlite()
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


class TestOntologyEnforcement(unittest.TestCase):
    """entity_type and relation_type are closed sets at the write boundary."""

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
        from tests.support import close_sqlite
        close_sqlite()
        for p in self.patches:
            p.stop()
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    # ---- entity types ----

    def test_unknown_entity_type_rejected_with_valid_list(self):
        from src.tools.entities import tool_create_entity
        result = tool_create_entity("Thing", "gizmo", "test")
        self.assertIn("Error", result)
        self.assertIn("gizmo", result)
        self.assertIn("person", result)  # the valid list is spelled out

    def test_entity_type_case_insensitive(self):
        from src.tools.entities import tool_create_entity
        result = tool_create_entity("Steve", "Person", "test")
        self.assertIn("Entity created", result)
        self.assertIn("(person)", result)

    def test_update_entity_rejects_unknown_type(self):
        from src.tools.entities import tool_create_entity, tool_update_entity
        tool_create_entity("Steve", "person", "test")
        result = tool_update_entity("Steve", new_type="wizard", vault="test")
        self.assertIn("Error", result)
        self.assertIn("wizard", result)

    # ---- relation types ----

    def _two_entities(self):
        from src.tools.entities import tool_create_entity
        tool_create_entity("A", "person", "test")
        tool_create_entity("B", "organization", "test")

    def test_canonical_relation_type_accepted(self):
        from src.tools.relations import tool_create_relation
        self._two_entities()
        result = tool_create_relation("A", "B", "works_at", vault="test")
        self.assertIn("Relation created", result)
        self.assertIn("works_at", result)

    def test_unknown_relation_type_rejected_with_valid_list(self):
        from src.tools.relations import tool_create_relation
        self._two_entities()
        result = tool_create_relation("A", "B", "death_catalyzed", vault="test")
        self.assertIn("Error", result)
        self.assertIn("death_catalyzed", result)
        self.assertIn("related_to", result)  # escape hatch is suggested

    def test_alias_is_canonicalized(self):
        from src.tools.relations import tool_create_relation
        self._two_entities()
        result = tool_create_relation("A", "B", "friends_with", vault="test")
        self.assertIn("Relation created", result)
        self.assertIn("[friend_of]", result)
        self.assertIn("canonicalized from 'friends_with'", result)

    def test_flipped_alias_swaps_endpoints(self):
        from src.tools.relations import tool_create_relation
        from src.graph.manager import get_all_relations
        from src.indexer.store import get_entity_by_name
        self._two_entities()

        # "A created_by B" states the same fact as "B created A".
        result = tool_create_relation("A", "B", "created_by", vault="test")
        self.assertIn("Relation created", result)
        self.assertIn("direction flipped", result)

        a = get_entity_by_name("A", "test")
        b = get_entity_by_name("B", "test")
        rel = get_all_relations()[0]
        self.assertEqual(rel.relation_type, "created")
        self.assertEqual(rel.from_entity, b.id)
        self.assertEqual(rel.to_entity, a.id)

    def test_rejected_relation_writes_nothing(self):
        from src.tools.relations import tool_create_relation
        from src.graph.manager import get_all_relations
        self._two_entities()
        tool_create_relation("A", "B", "nonsense_type", vault="test")
        self.assertEqual(get_all_relations(), [])


if __name__ == "__main__":
    unittest.main()
