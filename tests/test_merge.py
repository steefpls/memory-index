"""Tests for merge_entities: move observations + relations, soft-delete source,
re-embed under the target prefix, supersede-chain stability, export/import
survival, and search visibility under the target."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _make_test_patches(tmpdir):
    return [
        patch("src.config.DATA_DIR", Path(tmpdir)),
        patch("src.config.DB_FILE", Path(tmpdir) / "memory.db"),
        patch("src.config.CHROMA_DIR", Path(tmpdir) / "chroma"),
    ]


def _reset_state():
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
    # Explicit (not via auto-create): tool modules bind VAULTS at first
    # import, so a later `VAULTS = {}` rebind leaves them looking at a stale
    # dict that already contains 'test' — auto-create would then be skipped
    # while the live registry lacks it. Same pattern as test_store.py.
    config_mod.VAULTS["test"] = config_mod.VaultConfig(
        name="test", collection_name="memory_test")


class MergeTestCase(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.patches = _make_test_patches(self.tmpdir)

        self.mock_collection = MagicMock()
        self.mock_ef = MagicMock(return_value=[[0.1] * 768])
        self.patches.append(
            patch("src.indexer.store.get_collection",
                  return_value=self.mock_collection))
        self.patches.append(
            patch("src.indexer.store.get_embedding_function",
                  return_value=self.mock_ef))

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


class TestMergeEntities(MergeTestCase):
    def test_merge_moves_observations_and_keeps_metadata(self):
        from src.tools.entities import tool_create_entity, tool_merge_entities
        from src.indexer.store import get_observations, get_entity_by_name

        tool_create_entity("Source", "concept", "test")
        tool_create_entity("Target", "concept", "test")
        src = get_entity_by_name("Source", "test")
        tgt = get_entity_by_name("Target", "test")

        from src.indexer.store import add_observation
        o1 = add_observation(src.id, "Source fact one", source="chat 2026-09-01",
                             occurred_at="2026-08-01", created_at="2026-08-02T00:00:00+00:00")
        o2 = add_observation(src.id, "Source fact two", source="notes")
        o1_id, o2_id = o1.id, o2.id

        result = tool_merge_entities("Source", "Target", vault="test")
        self.assertIn("Merged", result)
        self.assertIn("2", result)

        # Source gone (soft-deleted -> invisible).
        self.assertIsNone(get_entity_by_name("Source", "test"))
        # Target holds both moved facts with IDs + metadata stable.
        by_id = {o.id: o for o in get_observations(tgt.id, include_superseded=True)}
        self.assertIn(o1_id, by_id)
        self.assertIn(o2_id, by_id)
        self.assertEqual(by_id[o1_id].content, "Source fact one")
        self.assertEqual(by_id[o1_id].source, "chat 2026-09-01")
        self.assertEqual(by_id[o1_id].occurred_at, "2026-08-01")
        self.assertEqual(by_id[o1_id].created_at, "2026-08-02T00:00:00+00:00")
        self.assertEqual(by_id[o2_id].source, "notes")
        # Source holds nothing anymore.
        self.assertEqual(
            [o for o in get_observations(src.id, include_superseded=True)], [])

    def test_merge_moves_relations_and_dedupes(self):
        from src.tools.entities import tool_create_entity, tool_merge_entities
        from src.tools.relations import tool_create_relation
        from src.graph.manager import get_all_relations
        from src.indexer.store import get_entity_by_name

        tool_create_entity("Src", "concept", "test")
        tool_create_entity("Dst", "concept", "test")
        tool_create_entity("Other", "concept", "test")

        # Dst -> Other already exists; Src -> Other will collide after re-point.
        tool_create_relation("Dst", "Other", "uses", vault="test")
        tool_create_relation("Src", "Other", "uses", vault="test")
        tool_create_relation("Other", "Src", "related_to", vault="test")

        result = tool_merge_entities("Src", "Dst", vault="test")
        self.assertIn("Merged", result)

        src = get_entity_by_name("Src", "test")
        self.assertIsNone(src)  # gone
        dst = get_entity_by_name("Dst", "test")
        other = get_entity_by_name("Other", "test")

        rels = get_all_relations()
        # No dangling reference to the deleted source.
        for r in rels:
            self.assertNotIn("Src", (r.from_entity, r.to_entity))
            self.assertNotEqual(r.from_entity, r.to_entity if False else "impossible")
        sigs = {(r.from_entity, r.to_entity, r.relation_type) for r in rels}
        # Dst->Other (uses) exists exactly once (duplicate removed).
        self.assertEqual(
            sum(1 for s in sigs if s == (dst.id, other.id, "uses")), 1)
        # Other->Dst (related_to) re-pointed from Other->Src.
        self.assertIn((other.id, dst.id, "related_to"), sigs)
        self.assertIn("duplicate", result.lower())

    def test_merge_reembeds_with_target_prefix_in_one_batch(self):
        from src.tools.entities import tool_create_entity, tool_merge_entities
        from src.indexer.store import get_entity_by_name

        tool_create_entity("Old", "person", "test")
        tool_create_entity("New", "project", "test")
        from src.indexer.store import add_observation
        src = get_entity_by_name("Old", "test")
        add_observation(src.id, "fact A")
        add_observation(src.id, "fact B")

        self.mock_collection.upsert.reset_mock()
        tool_merge_entities("Old", "New", vault="test")

        self.assertTrue(self.mock_collection.upsert.called)
        kwargs = self.mock_collection.upsert.call_args[1]
        self.assertEqual(len(kwargs["ids"]), 2)
        for doc in kwargs["documents"]:
            self.assertTrue(doc.startswith("project: New\n"),
                            f"expected target prefix, got: {doc[:40]!r}")
        for meta in kwargs["metadatas"]:
            tgt = get_entity_by_name("New", "test")
            self.assertEqual(meta["entity_id"], tgt.id)
            self.assertEqual(meta["entity_type"], "project")

    def test_merge_preserves_supersede_chain(self):
        from src.tools.entities import tool_create_entity, tool_merge_entities
        from src.indexer.store import add_observation, get_observations, get_entity_by_name

        tool_create_entity("S", "concept", "test")
        tool_create_entity("T", "concept", "test")
        src = get_entity_by_name("S", "test")
        v1 = add_observation(src.id, "v1 fact")
        v2 = add_observation(src.id, "v2 fact", supersedes=v1.id)

        tool_merge_entities("S", "T", vault="test")

        tgt = get_entity_by_name("T", "test")
        active = get_observations(tgt.id)
        self.assertEqual([o.content for o in active], ["v2 fact"])
        all_obs = {o.id: o for o in get_observations(tgt.id, include_superseded=True)}
        self.assertIn(v1.id, all_obs)
        self.assertEqual(all_obs[v1.id].superseded_by, v2.id)

    def test_merge_rejects_unknown_and_cross_vault(self):
        from src.tools.entities import tool_create_entity, tool_merge_entities

        tool_create_entity("A", "concept", "test")
        tool_create_entity("B", "concept", "test")
        self.assertIn("not found", tool_merge_entities("Nope", "B", vault="test"))
        self.assertIn("not found", tool_merge_entities("A", "Nope", vault="test"))
        self.assertIn("itself", tool_merge_entities("A", "A", vault="test").lower())

        import src.config as config_mod
        config_mod.VAULTS["other"] = config_mod.VaultConfig(
            name="other", collection_name="memory_other")
        tool_create_entity("C", "concept", "other")
        result = tool_merge_entities("A", "C")
        self.assertIn("Error", result)
        self.assertIn("vault", result.lower())

    def test_search_finds_moved_facts_under_target(self):
        """Vector search joins hits back to the store row — after a merge the
        same observation ID must resolve under the TARGET entity."""
        import json as _json
        from src.tools.entities import tool_create_entity, tool_merge_entities
        from src.indexer.store import add_observation, get_entity_by_name
        from src.tools import search as search_mod

        tool_create_entity("SrcE", "concept", "test")
        tool_create_entity("DstE", "concept", "test")
        src = get_entity_by_name("SrcE", "test")
        moved = add_observation(src.id, "quokka habitat uniquely marsupial")
        tool_merge_entities("SrcE", "DstE", vault="test")

        class _FakeColl:
            def query(self, query_embeddings=None, n_results=10, where=None,
                      include=None):
                return {"ids": [[moved.id]], "distances": [[0.1]]}

        search_mod._calibration_cache["test"] = {
            "HIGH": 0.6, "MEDIUM": 1.0, "LOW": 1.4}
        try:
            with patch.object(search_mod, "VAULTS", {"test": object()}), \
                 patch("src.tools.search.get_vault",
                       return_value=MagicMock(collection_name="test")), \
                 patch("src.tools.search.get_collection",
                       return_value=_FakeColl()), \
                 patch("src.tools.search._get_query_embeddings_with_guard",
                       return_value=[[0.1] * 8]):
                payload = _json.loads(search_mod.search_memory(
                    "quokka habitat", vault="test", output_format="json"))
        finally:
            search_mod._calibration_cache.clear()

        self.assertEqual(len(payload["results"]), 1)
        self.assertEqual(payload["results"][0]["content"],
                         "quokka habitat uniquely marsupial")
        self.assertEqual(payload["results"][0]["entity_name"], "DstE")

    def test_merge_survives_export_import(self):
        from src.tools.entities import tool_create_entity, tool_merge_entities
        from src.tools.relations import tool_create_relation
        from src.tools.portability import tool_export_vault, tool_import_vault
        from src.indexer.store import get_entity_by_name, get_observations
        from src.graph.manager import get_all_relations
        import src.config as config_mod

        tool_create_entity("OldOne", "concept", "test")
        tool_create_entity("NewOne", "concept", "test")
        tool_create_entity("Third", "concept", "test")
        from src.indexer.store import add_observation
        old = get_entity_by_name("OldOne", "test")
        add_observation(old.id, "merged fact 1")
        v1 = add_observation(old.id, "versioned fact v1")
        add_observation(old.id, "versioned fact v2", supersedes=v1.id)
        tool_create_relation("OldOne", "Third", "uses", vault="test")

        tool_merge_entities("OldOne", "NewOne", vault="test")

        out = str(Path(self.tmpdir) / "merged.zip")
        # Point the portability DATA_DIR at the tmpdir for the export target.
        with patch("src.tools.portability.DATA_DIR", Path(self.tmpdir)):
            tool_export_vault("test", out)

        config_mod.VAULTS["restored"] = config_mod.VaultConfig(
            name="restored", collection_name="memory_restored")
        tool_import_vault(out, "restored")

        # Source stays gone; target owns the facts incl. history.
        self.assertIsNone(get_entity_by_name("OldOne", "restored"))
        new = get_entity_by_name("NewOne", "restored")
        self.assertIsNotNone(new)
        contents = {o.content for o in get_observations(
            new.id, include_superseded=True)}
        self.assertEqual(contents, {"merged fact 1", "versioned fact v1",
                                    "versioned fact v2"})
        active = {o.content for o in get_observations(new.id)}
        self.assertEqual(active, {"merged fact 1", "versioned fact v2"})
        # Relation re-pointed at the target in the restored vault.
        third = get_entity_by_name("Third", "restored")
        sigs = {(r.from_entity, r.to_entity, r.relation_type)
                for r in get_all_relations()
                if r.from_entity in (new.id, third.id)
                and r.to_entity in (new.id, third.id)}
        self.assertIn((new.id, third.id, "uses"), sigs)


if __name__ == "__main__":
    unittest.main()
