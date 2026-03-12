"""Tests for entity/observation store."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestEntityStore(unittest.TestCase):
    """Test entity and observation CRUD operations."""

    def setUp(self):
        """Set up test fixtures with temporary data directory."""
        self.tmpdir = tempfile.mkdtemp()
        self.patches = []

        # Patch config paths to use temp dir
        p1 = patch("src.config.DATA_DIR", Path(self.tmpdir))
        p2 = patch("src.config.ENTITIES_FILE", Path(self.tmpdir) / "memory_entities.json")
        p3 = patch("src.config.VAULTS_FILE", Path(self.tmpdir) / "vaults.json")
        p4 = patch("src.config.GRAPH_FILE", Path(self.tmpdir) / "memory_graph.json")
        p5 = patch("src.config.CHROMA_DIR", Path(self.tmpdir) / "chroma")
        self.patches.extend([p1, p2, p3, p4, p5])
        for p in self.patches:
            p.start()

        # Also patch the store module's imported references
        p6 = patch("src.indexer.store.DATA_DIR", Path(self.tmpdir))
        p7 = patch("src.indexer.store.ENTITIES_FILE", Path(self.tmpdir) / "memory_entities.json")
        self.patches.extend([p6, p7])
        p6.start()
        p7.start()

        # Reset store state
        import src.indexer.store as store_mod
        store_mod._entities = {}
        store_mod._observations = {}
        store_mod._loaded = True  # skip file load

        # Create a test vault
        import src.config as config_mod
        config_mod.VAULTS = {}
        config_mod.VAULTS["test"] = config_mod.VaultConfig(name="test", collection_name="memory_test")

        # Mock ChromaDB collection
        self.mock_collection = MagicMock()
        self.mock_collection.add = MagicMock()
        self.mock_collection.delete = MagicMock()
        self.mock_collection.upsert = MagicMock()
        p8 = patch("src.indexer.store.get_collection", return_value=self.mock_collection)
        self.patches.append(p8)
        p8.start()

        # Mock embedding function
        self.mock_ef = MagicMock()
        self.mock_ef.__call__ = MagicMock(return_value=[[0.1] * 768])
        p9 = patch("src.indexer.store.get_embedding_function", return_value=self.mock_ef)
        self.patches.append(p9)
        p9.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_create_entity(self):
        from src.indexer.store import create_entity, get_entity

        entity = create_entity("Python", "technology", "test")
        self.assertEqual(entity.name, "Python")
        self.assertEqual(entity.entity_type, "technology")
        self.assertEqual(entity.vault, "test")
        self.assertFalse(entity.deleted)

        # Retrieve by ID
        retrieved = get_entity(entity.id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.name, "Python")

    def test_create_entity_with_observations(self):
        from src.indexer.store import create_entity, get_observations

        entity = create_entity("Python", "technology", "test",
                              observations=["General purpose language", "Created by Guido"])

        obs = get_observations(entity.id)
        self.assertEqual(len(obs), 2)
        self.assertEqual(obs[0].content, "General purpose language")
        self.assertEqual(obs[1].content, "Created by Guido")

        # Verify ChromaDB was called
        self.assertEqual(self.mock_collection.add.call_count, 2)

    def test_create_entity_idempotent(self):
        from src.indexer.store import create_entity

        e1 = create_entity("Python", "technology", "test")
        e2 = create_entity("Python", "technology", "test")
        self.assertEqual(e1.id, e2.id)

    def test_get_entity_by_name(self):
        from src.indexer.store import create_entity, get_entity_by_name

        create_entity("Python", "technology", "test")
        ent = get_entity_by_name("Python", "test")
        self.assertIsNotNone(ent)
        self.assertEqual(ent.name, "Python")

    def test_update_entity(self):
        from src.indexer.store import create_entity, update_entity

        entity = create_entity("Pytohn", "technology", "test")
        updated = update_entity(entity.id, name="Python")
        self.assertEqual(updated.name, "Python")

    def test_delete_entity(self):
        from src.indexer.store import create_entity, delete_entity, get_entity

        entity = create_entity("Python", "technology", "test")
        result = delete_entity(entity.id)
        self.assertTrue(result)
        self.assertIsNone(get_entity(entity.id))

    def test_delete_entity_removes_observations(self):
        from src.indexer.store import create_entity, delete_entity, get_observations

        entity = create_entity("Python", "technology", "test",
                              observations=["A fact"])
        delete_entity(entity.id)
        obs = get_observations(entity.id)
        self.assertEqual(len(obs), 0)

    def test_list_entities(self):
        from src.indexer.store import create_entity, list_entities

        create_entity("Python", "technology", "test")
        create_entity("Alice", "person", "test")

        entities, total = list_entities(vault="test")
        self.assertEqual(total, 2)
        self.assertEqual(len(entities), 2)

    def test_list_entities_filter_type(self):
        from src.indexer.store import create_entity, list_entities

        create_entity("Python", "technology", "test")
        create_entity("Alice", "person", "test")

        entities, total = list_entities(vault="test", entity_type="person")
        self.assertEqual(total, 1)
        self.assertEqual(entities[0].name, "Alice")

    def test_add_observation(self):
        from src.indexer.store import create_entity, add_observation, get_observations

        entity = create_entity("Python", "technology", "test")
        obs = add_observation(entity.id, "Used in ML")
        self.assertIsNotNone(obs)
        self.assertEqual(obs.content, "Used in ML")

        all_obs = get_observations(entity.id)
        self.assertEqual(len(all_obs), 1)

    def test_delete_observation(self):
        from src.indexer.store import create_entity, add_observation, delete_observation, get_observations

        entity = create_entity("Python", "technology", "test")
        obs = add_observation(entity.id, "A fact")
        result = delete_observation(obs.id)
        self.assertTrue(result)
        self.assertEqual(len(get_observations(entity.id)), 0)

    def test_resolve_entity_by_id(self):
        from src.indexer.store import create_entity, resolve_entity

        entity = create_entity("Python", "technology", "test")
        resolved = resolve_entity(entity.id)
        self.assertIsNotNone(resolved)
        self.assertEqual(resolved.name, "Python")

    def test_resolve_entity_by_name(self):
        from src.indexer.store import create_entity, resolve_entity

        create_entity("Python", "technology", "test")
        resolved = resolve_entity("Python", "test")
        self.assertIsNotNone(resolved)

    def test_entity_count(self):
        from src.indexer.store import create_entity, get_entity_count

        create_entity("Python", "technology", "test")
        create_entity("Rust", "technology", "test")
        self.assertEqual(get_entity_count(vault="test"), 2)

    def test_observation_count(self):
        from src.indexer.store import create_entity, add_observation, get_observation_count

        entity = create_entity("Python", "technology", "test")
        add_observation(entity.id, "Fact 1")
        add_observation(entity.id, "Fact 2")
        self.assertEqual(get_observation_count(vault="test"), 2)

    # --- Superseding tests ---

    def test_supersede_observation(self):
        """Superseding marks old observation and links to new one."""
        from src.indexer.store import create_entity, add_observation, get_observations

        entity = create_entity("Perception", "project", "test")
        old = add_observation(entity.id, "Uses .NET Framework")
        new = add_observation(entity.id, "Migrated to .NET 8", supersedes=old.id)

        # Default: only current observations
        current = get_observations(entity.id)
        self.assertEqual(len(current), 1)
        self.assertEqual(current[0].content, "Migrated to .NET 8")

        # Include superseded: both show up
        all_obs = get_observations(entity.id, include_superseded=True)
        self.assertEqual(len(all_obs), 2)

    def test_superseded_observation_has_link(self):
        """Superseded observation stores the ID of its replacement."""
        from src.indexer.store import create_entity, add_observation, _observations

        entity = create_entity("Perception", "project", "test")
        old = add_observation(entity.id, "Uses .NET Framework")
        new = add_observation(entity.id, "Migrated to .NET 8", supersedes=old.id)

        old_obs = _observations[old.id]
        self.assertEqual(old_obs.superseded_by, new.id)
        self.assertTrue(old_obs.is_superseded)

    def test_supersede_updates_chromadb_metadata(self):
        """Superseding should update ChromaDB metadata, not delete the entry."""
        from src.indexer.store import create_entity, add_observation

        entity = create_entity("Perception", "project", "test")
        old = add_observation(entity.id, "Uses .NET Framework")
        new = add_observation(entity.id, "Migrated to .NET 8", supersedes=old.id)

        # ChromaDB update should have been called (not delete)
        self.mock_collection.update.assert_called()
        update_call = self.mock_collection.update.call_args
        meta = update_call[1]["metadatas"][0] if "metadatas" in update_call[1] else update_call[0][1][0]
        self.assertEqual(meta.get("superseded_by"), new.id)

    def test_superseded_excluded_from_count(self):
        """Observation count should not include superseded observations."""
        from src.indexer.store import create_entity, add_observation, get_observation_count

        entity = create_entity("Perception", "project", "test")
        old = add_observation(entity.id, "Uses .NET Framework")
        add_observation(entity.id, "Migrated to .NET 8", supersedes=old.id)

        self.assertEqual(get_observation_count(vault="test"), 1)

    def test_supersede_wrong_entity_ignored(self):
        """Superseding an observation from a different entity should be ignored."""
        from src.indexer.store import create_entity, add_observation, get_observations

        entity_a = create_entity("A", "project", "test")
        entity_b = create_entity("B", "project", "test")
        obs_a = add_observation(entity_a.id, "Fact for A")
        add_observation(entity_b.id, "Fact for B", supersedes=obs_a.id)

        # obs_a should NOT be superseded since it belongs to a different entity
        current_a = get_observations(entity_a.id)
        self.assertEqual(len(current_a), 1)
        self.assertFalse(current_a[0].is_superseded)

    def test_supersede_chain(self):
        """Multiple supersedes in a chain should only leave the latest current."""
        from src.indexer.store import create_entity, add_observation, get_observations

        entity = create_entity("Framework", "technology", "test")
        v1 = add_observation(entity.id, ".NET Framework")
        v2 = add_observation(entity.id, ".NET 8", supersedes=v1.id)
        v3 = add_observation(entity.id, ".NET 12", supersedes=v2.id)

        current = get_observations(entity.id)
        self.assertEqual(len(current), 1)
        self.assertEqual(current[0].content, ".NET 12")

        all_obs = get_observations(entity.id, include_superseded=True)
        self.assertEqual(len(all_obs), 3)

    # --- Temporal metadata tests ---

    def test_observation_has_created_at_in_chromadb(self):
        """ChromaDB metadata should include created_at timestamp."""
        from src.indexer.store import create_entity, add_observation

        entity = create_entity("Python", "technology", "test")
        obs = add_observation(entity.id, "A fact")

        add_call = self.mock_collection.add.call_args
        meta = add_call[1]["metadatas"][0] if "metadatas" in add_call[1] else add_call[0][3][0]
        self.assertIn("created_at", meta)
        self.assertEqual(meta["created_at"], obs.created_at)

    def test_observation_serialization_with_superseded(self):
        """Observation to_dict/from_dict should roundtrip superseded_by."""
        from src.models.observation import Observation

        obs = Observation(id="abc", entity_id="xyz", content="test",
                          superseded_by="def")
        d = obs.to_dict()
        self.assertEqual(d["superseded_by"], "def")

        restored = Observation.from_dict(d)
        self.assertEqual(restored.superseded_by, "def")
        self.assertTrue(restored.is_superseded)

    def test_observation_serialization_without_superseded(self):
        """Observation without superseded_by should omit it from dict."""
        from src.models.observation import Observation

        obs = Observation(id="abc", entity_id="xyz", content="test")
        d = obs.to_dict()
        self.assertNotIn("superseded_by", d)

        restored = Observation.from_dict(d)
        self.assertEqual(restored.superseded_by, "")
        self.assertFalse(restored.is_superseded)


if __name__ == "__main__":
    unittest.main()
