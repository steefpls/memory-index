"""Tests for entity/observation store."""

import json
import os
import sys
import tempfile
import threading
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

        # Point the SQLite store at the temp dir
        from tests.support import patch_sqlite
        self.db_mod = patch_sqlite(self.tmpdir, self.patches)

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
        from tests.support import close_sqlite
        close_sqlite()
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

        # Initial observations go through the batch path: 2 facts, but only
        # ONE Chroma add (and one embedder call) for the whole batch.
        self.assertEqual(self.mock_collection.add.call_count, 1)
        add_call = self.mock_collection.add.call_args
        self.assertEqual(len(add_call[1]["ids"]), 2)

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

    def test_supersede_records_pointer_and_timestamp_in_store(self):
        """Superseding is a store-only mutation: the old row gets the pointer
        plus a superseded_at stamp, and Chroma is neither updated nor deleted
        (the vector stays searchable; search joins back to the store row)."""
        from src.indexer.store import create_entity, add_observation

        entity = create_entity("Perception", "project", "test")
        old = add_observation(entity.id, "Uses .NET Framework")
        new = add_observation(entity.id, "Migrated to .NET 8", supersedes=old.id)

        self.assertEqual(old.superseded_by, new.id)
        self.assertTrue(old.superseded_at)
        self.mock_collection.update.assert_not_called()
        self.mock_collection.delete.assert_not_called()

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

    def test_chroma_metadata_is_minimal(self):
        """Chroma metadata carries ONLY the keys its where-filters need.

        Content, source, and timestamps live in the SQLite store; duplicating
        them into Chroma is what let the two stores hold divergent copies of a
        fact.
        """
        from src.indexer.store import create_entity, add_observation

        entity = create_entity("Python", "technology", "test")
        add_observation(entity.id, "A fact")

        add_call = self.mock_collection.add.call_args
        meta = add_call[1]["metadatas"][0] if "metadatas" in add_call[1] else add_call[0][3][0]
        self.assertEqual(set(meta.keys()), {"entity_id", "entity_type"})
        self.assertEqual(meta["entity_id"], entity.id)
        self.assertEqual(meta["entity_type"], "technology")

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

    # --- Batch write tests ---

    def test_add_observations_single_embed_and_add(self):
        """A batch of N facts costs ONE embedder call and ONE Chroma add."""
        from src.indexer.store import create_entity, add_observations, get_observations

        entity = create_entity("Python", "technology", "test")
        self.mock_collection.add.reset_mock()
        self.mock_ef.reset_mock()

        created = add_observations(entity.id, ["Fact A", "Fact B", "Fact C"])

        self.assertEqual(len(created), 3)
        self.assertEqual(self.mock_collection.add.call_count, 1)
        self.assertEqual(self.mock_ef.call_count, 1)

        # The one embedder call received the whole list of texts.
        embed_arg = self.mock_ef.call_args[0][0]
        self.assertEqual(len(embed_arg), 3)

        contents = {o.content for o in get_observations(entity.id)}
        self.assertEqual(contents, {"Fact A", "Fact B", "Fact C"})

    def test_add_observations_one_db_write_per_batch(self):
        """The batch path persists observations in one DB call, not one per fact."""
        import src.indexer.store as store_mod
        from src.indexer.store import create_entity, add_observations

        entity = create_entity("Python", "technology", "test")
        with patch.object(store_mod.db, "upsert_observations",
                          wraps=store_mod.db.upsert_observations) as spy:
            add_observations(entity.id, ["A", "B", "C", "D"])
        self.assertEqual(spy.call_count, 1)
        self.assertEqual(len(spy.call_args[0][0]), 4)

    def test_add_observations_preserves_order_and_blank_filtering(self):
        from src.indexer.store import create_entity, add_observations

        entity = create_entity("Python", "technology", "test")
        created = add_observations(entity.id, ["one", "  ", "two", ""])
        self.assertEqual([o.content for o in created], ["one", "two"])

    def test_add_observations_unknown_entity(self):
        from src.indexer.store import add_observations
        self.assertEqual(add_observations("nope", ["x"]), [])

    def test_add_observations_empty_list(self):
        from src.indexer.store import create_entity, add_observations

        entity = create_entity("Python", "technology", "test")
        self.assertEqual(add_observations(entity.id, []), [])

    def test_add_observations_pipe_survives(self):
        """A fact containing '|' must be stored verbatim, never split."""
        from src.indexer.store import create_entity, add_observations, get_observations

        entity = create_entity("Python", "technology", "test")
        fact = "Pipeline is stdin | grep | wc -l"
        add_observations(entity.id, [fact])

        obs = get_observations(entity.id)
        self.assertEqual(len(obs), 1)
        self.assertEqual(obs[0].content, fact)

    # --- Auto-recalibration triggers (must survive batched writes) ---

    def test_batch_write_crossing_the_boundary_recalibrates(self):
        """A batch that steps OVER the every-10 boundary must still trigger.

        The old modulo test only fired on an exact landing, so 7 + 5 = 12
        skipped the trigger permanently.
        """
        import src.indexer.store as store_mod
        from src.indexer.store import create_entity, add_observations

        entity = create_entity("Python", "technology", "test")
        add_observations(entity.id, [f"f{i}" for i in range(7)])

        with patch.object(store_mod, "calibrate_collection") as cal, \
             patch.object(store_mod, "_LIBRARIAN_EVERY", 10_000):
            add_observations(entity.id, [f"g{i}" for i in range(5)])  # 7 -> 12

        cal.assert_called_once()

    def test_boundary_fires_exactly_once(self):
        """Crossing 10 fires; the next write inside the same decade does not."""
        import src.indexer.store as store_mod
        from src.indexer.store import create_entity, add_observation, add_observations

        entity = create_entity("Python", "technology", "test")
        add_observations(entity.id, [f"f{i}" for i in range(12)])  # 0 -> 12

        with patch.object(store_mod, "calibrate_collection") as cal, \
             patch.object(store_mod, "_LIBRARIAN_EVERY", 10_000):
            add_observation(entity.id, "one more")  # 12 -> 13
        cal.assert_not_called()

    def test_single_writes_still_trigger_on_the_boundary(self):
        import src.indexer.store as store_mod
        from src.indexer.store import create_entity, add_observation

        entity = create_entity("Python", "technology", "test")
        for i in range(9):
            add_observation(entity.id, f"f{i}")

        with patch.object(store_mod, "calibrate_collection") as cal, \
             patch.object(store_mod, "_LIBRARIAN_EVERY", 10_000):
            add_observation(entity.id, "the tenth")  # 9 -> 10
        cal.assert_called_once()

    def test_crossed_multiple_helper(self):
        from src.indexer.store import _crossed_multiple

        self.assertTrue(_crossed_multiple(7, 12, 10))
        self.assertTrue(_crossed_multiple(9, 10, 10))
        self.assertTrue(_crossed_multiple(0, 25, 10))
        self.assertFalse(_crossed_multiple(10, 11, 10))
        self.assertFalse(_crossed_multiple(12, 13, 10))
        self.assertFalse(_crossed_multiple(12, 11, 10))  # supersession shrank it
        self.assertFalse(_crossed_multiple(0, 0, 10))

    def test_add_observations_mismatched_occurred_at_raises(self):
        from src.indexer.store import create_entity, add_observations

        entity = create_entity("Python", "technology", "test")
        with self.assertRaises(ValueError):
            add_observations(entity.id, ["a", "b"], occurred_at=["2026-01-01"])

    # --- occurred_at tests ---

    def test_add_observation_occurred_at_stored(self):
        from src.indexer.store import create_entity, add_observation

        entity = create_entity("Python", "technology", "test")
        obs = add_observation(entity.id, "Released 3.13", occurred_at="2024-10-07")
        self.assertEqual(obs.occurred_at, "2024-10-07")

    def test_add_observation_occurred_at_defaults_none(self):
        from src.indexer.store import create_entity, add_observation

        entity = create_entity("Python", "technology", "test")
        obs = add_observation(entity.id, "A fact")
        self.assertIsNone(obs.occurred_at)

    def test_occurred_at_lives_in_store_not_chroma(self):
        """occurred_at is store data; Chroma metadata stays minimal."""
        from src.indexer.store import create_entity, add_observation

        entity = create_entity("Python", "technology", "test")
        self.mock_collection.add.reset_mock()
        obs = add_observation(entity.id, "A fact", occurred_at="2025-05-01")

        self.assertEqual(obs.occurred_at, "2025-05-01")
        meta = self.mock_collection.add.call_args[1]["metadatas"][0]
        self.assertNotIn("occurred_at", meta)

    def test_add_observations_parallel_occurred_at(self):
        from src.indexer.store import create_entity, add_observations

        entity = create_entity("Python", "technology", "test")
        created = add_observations(
            entity.id, ["First", "Second"],
            occurred_at=["2024-01-01", "2025-02-02"],
        )
        self.assertEqual(created[0].occurred_at, "2024-01-01")
        self.assertEqual(created[1].occurred_at, "2025-02-02")

    def test_add_observations_partial_occurred_at(self):
        """Blank entries in the occurred_at list mean 'unknown', not ''."""
        from src.indexer.store import create_entity, add_observations

        entity = create_entity("Python", "technology", "test")
        created = add_observations(entity.id, ["First", "Second"],
                                   occurred_at=["2024-01-01", ""])
        self.assertEqual(created[0].occurred_at, "2024-01-01")
        self.assertIsNone(created[1].occurred_at)

    def test_observation_occurred_at_roundtrip(self):
        from src.models.observation import Observation

        obs = Observation(id="a", entity_id="b", content="c",
                          occurred_at="2026-03-13")
        d = obs.to_dict()
        self.assertEqual(d["occurred_at"], "2026-03-13")
        self.assertEqual(Observation.from_dict(d).occurred_at, "2026-03-13")

    def test_observation_occurred_at_omitted_when_unset(self):
        from src.models.observation import Observation

        d = Observation(id="a", entity_id="b", content="c").to_dict()
        self.assertNotIn("occurred_at", d)
        self.assertIsNone(Observation.from_dict(d).occurred_at)

    def test_effective_at_prefers_occurred_at(self):
        from src.models.observation import Observation

        with_event = Observation(id="a", entity_id="b", content="c",
                                 created_at="2026-08-01T00:00:00+00:00",
                                 occurred_at="2020-01-01")
        self.assertEqual(with_event.effective_at, "2020-01-01")

        without = Observation(id="a", entity_id="b", content="c",
                              created_at="2026-08-01T00:00:00+00:00")
        self.assertEqual(without.effective_at, "2026-08-01T00:00:00+00:00")

    # --- mark_superseded ---

    def test_mark_superseded(self):
        from src.indexer.store import (
            create_entity, add_observation, mark_superseded, get_observations,
        )

        entity = create_entity("Python", "technology", "test")
        old = add_observation(entity.id, "Old")
        new = add_observation(entity.id, "New")

        self.assertTrue(mark_superseded(old.id, new.id))
        active = get_observations(entity.id)
        self.assertEqual([o.content for o in active], ["New"])

    def test_mark_superseded_tolerates_unresolved_pointer(self):
        """A dangling pointer still takes the row out of active reads."""
        from src.indexer.store import (
            create_entity, add_observation, mark_superseded, get_observations,
        )

        entity = create_entity("Python", "technology", "test")
        old = add_observation(entity.id, "Old")

        self.assertTrue(mark_superseded(old.id, "not-a-local-id"))
        self.assertEqual(len(get_observations(entity.id)), 0)
        self.assertEqual(len(get_observations(entity.id, include_superseded=True)), 1)

    def test_mark_superseded_unknown_observation(self):
        from src.indexer.store import mark_superseded
        self.assertFalse(mark_superseded("nope", "also-nope"))


class TestAtomicPersistence(unittest.TestCase):
    """The store and graph must never leave a half-written JSON file behind."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_atomic_write_creates_file(self):
        from src.indexer.store import atomic_write_json

        target = Path(self.tmpdir) / "out.json"
        atomic_write_json(target, {"a": 1})
        self.assertEqual(json.loads(target.read_text(encoding="utf-8")), {"a": 1})

    def test_atomic_write_replaces_existing(self):
        from src.indexer.store import atomic_write_json

        target = Path(self.tmpdir) / "out.json"
        atomic_write_json(target, {"v": 1})
        atomic_write_json(target, {"v": 2})
        self.assertEqual(json.loads(target.read_text(encoding="utf-8")), {"v": 2})

    def test_atomic_write_leaves_no_temp_files(self):
        from src.indexer.store import atomic_write_json

        target = Path(self.tmpdir) / "out.json"
        atomic_write_json(target, {"a": 1})
        leftovers = [p.name for p in Path(self.tmpdir).iterdir()
                     if p.name != "out.json"]
        self.assertEqual(leftovers, [])

    def test_failed_write_preserves_old_file_and_cleans_up(self):
        """If serialization/IO blows up mid-write, the old file survives."""
        from src.indexer.store import atomic_write_json

        target = Path(self.tmpdir) / "out.json"
        atomic_write_json(target, {"good": True})

        class Boom:
            pass

        with self.assertRaises(TypeError):
            atomic_write_json(target, {"bad": Boom()})

        # Old content intact...
        self.assertEqual(json.loads(target.read_text(encoding="utf-8")),
                         {"good": True})
        # ...and no orphaned temp file left in the directory.
        leftovers = [p.name for p in Path(self.tmpdir).iterdir()
                     if p.name != "out.json"]
        self.assertEqual(leftovers, [])

    def test_temp_file_is_created_in_target_directory(self):
        """os.replace is only atomic within a volume, so the temp file must
        be a sibling of the target — not in the system temp dir."""
        import src.indexer.store as store_mod

        target = Path(self.tmpdir) / "out.json"
        seen = {}
        real_mkstemp = store_mod.tempfile.mkstemp

        def spy(*args, **kwargs):
            seen["dir"] = kwargs.get("dir")
            return real_mkstemp(*args, **kwargs)

        with patch.object(store_mod.tempfile, "mkstemp", side_effect=spy):
            store_mod.atomic_write_json(target, {"a": 1})

        self.assertEqual(Path(seen["dir"]), Path(self.tmpdir))

    def test_graph_save_persists_to_sqlite(self):
        """add_relation writes a durable row that a fresh load can read back."""
        import src.graph.manager as gm
        from src.models.relation import Relation
        from tests.support import patch_sqlite, close_sqlite

        patches = []
        db_mod = patch_sqlite(self.tmpdir, patches)
        try:
            gm._graph = None
            gm._relations = {}
            gm.add_relation(Relation(id="r1", from_entity="a", to_entity="b",
                                     relation_type="uses"))

            rows = db_mod.load_relations()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["id"], "r1")
            self.assertEqual(rows[0]["relation_type"], "uses")
        finally:
            gm._graph = None
            gm._relations = {}
            close_sqlite()
            for p in patches:
                p.stop()


class TestStoreConcurrency(unittest.TestCase):
    """Concurrent writers must not corrupt the in-memory dicts."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.patches = [
            patch("src.indexer.store.get_collection", return_value=MagicMock()),
            patch("src.indexer.store.get_embedding_function",
                  return_value=MagicMock(return_value=[[0.1] * 768])),
        ]
        for p in self.patches:
            p.start()

        from tests.support import patch_sqlite
        patch_sqlite(self.tmpdir, self.patches)

        import src.indexer.store as store_mod
        store_mod._entities = {}
        store_mod._observations = {}
        store_mod._loaded = True

        import src.config as config_mod
        config_mod.VAULTS = {
            "test": config_mod.VaultConfig(name="test", collection_name="memory_test"),
        }

    def tearDown(self):
        from tests.support import close_sqlite
        close_sqlite()
        for p in self.patches:
            p.stop()
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_store_lock_is_reentrant(self):
        """Nested store calls (create_entity -> add_observations) must not
        self-deadlock, which requires an RLock rather than a plain Lock."""
        import src.indexer.store as store_mod
        self.assertIsInstance(store_mod.STORE_LOCK, type(threading.RLock()))

        with store_mod.STORE_LOCK:
            with store_mod.STORE_LOCK:
                pass  # would hang on a non-reentrant lock

    def test_concurrent_writes_do_not_lose_observations(self):
        import threading as _threading
        from src.indexer.store import create_entity, add_observation, get_observations

        entity = create_entity("Python", "technology", "test")
        errors = []

        def worker(n):
            try:
                for i in range(20):
                    add_observation(entity.id, f"w{n}-fact{i}")
            except Exception as e:  # pragma: no cover - failure path
                errors.append(e)

        threads = [_threading.Thread(target=worker, args=(n,)) for n in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [])
        self.assertEqual(len(get_observations(entity.id)), 80)

    def test_concurrent_readers_never_see_a_mutating_dict(self):
        """Full-dict scans run under the lock, so a reader can't trip over a
        'dict changed size during iteration' RuntimeError."""
        import threading as _threading
        from src.indexer.store import (
            create_entity, add_observation, get_observation_count, list_entities,
        )

        entity = create_entity("Python", "technology", "test")
        stop = _threading.Event()
        errors = []

        def writer():
            try:
                for i in range(150):
                    add_observation(entity.id, f"fact{i}")
            finally:
                stop.set()

        def reader():
            try:
                while not stop.is_set():
                    get_observation_count("test")
                    list_entities(vault="test")
            except Exception as e:  # pragma: no cover - failure path
                errors.append(e)

        threads = [_threading.Thread(target=writer)] + [
            _threading.Thread(target=reader) for _ in range(3)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [])

    def test_concurrent_create_entity_is_idempotent(self):
        """Racing creates of the same name must yield exactly one entity."""
        import threading as _threading
        from src.indexer.store import create_entity, list_entities

        barrier = _threading.Barrier(6)
        results = []
        lock = _threading.Lock()

        def worker():
            barrier.wait()
            ent = create_entity("Racy", "concept", "test")
            with lock:
                results.append(ent.id)

        threads = [_threading.Thread(target=worker) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(set(results)), 1)
        entities, total = list_entities(vault="test")
        self.assertEqual(total, 1)


if __name__ == "__main__":
    unittest.main()
