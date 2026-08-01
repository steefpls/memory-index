"""Tests for vault import/export (portability)."""

import json
import os
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestPortability(unittest.TestCase):
    """Roundtrip tests for export_vault / import_vault."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.patches = []

        tmp = Path(self.tmpdir)

        # Patch all storage paths to the temp dir
        path_patches = [
            ("src.config.DATA_DIR", tmp),
            ("src.config.CHROMA_DIR", tmp / "chroma"),
            ("src.tools.portability.DATA_DIR", tmp),
        ]
        for target, value in path_patches:
            p = patch(target, value)
            self.patches.append(p)
            p.start()

        from tests.support import patch_sqlite
        patch_sqlite(self.tmpdir, self.patches)

        # Reset module state
        import src.indexer.store as store_mod
        store_mod._entities = {}
        store_mod._observations = {}
        store_mod._loaded = True

        import src.graph.manager as gm
        gm._graph = None
        gm._relations = {}

        # Vault registry — start with one source vault
        import src.config as config_mod
        config_mod.VAULTS = {
            "alpha": config_mod.VaultConfig(name="alpha", collection_name="memory_alpha"),
        }

        # Mock ChromaDB so we don't need a real backend
        self.mock_collection = MagicMock()
        p = patch("src.indexer.store.get_collection", return_value=self.mock_collection)
        self.patches.append(p)
        p.start()

        self.mock_ef = MagicMock(return_value=[[0.1] * 768])
        p = patch("src.indexer.store.get_embedding_function", return_value=self.mock_ef)
        self.patches.append(p)
        p.start()

    def tearDown(self):
        from tests.support import close_sqlite
        close_sqlite()
        for p in self.patches:
            p.stop()
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    # ----- helpers -----

    def _seed_alpha(self):
        """Create a small graph in vault 'alpha' for export."""
        from src.indexer.store import create_entity, add_observation
        from src.graph.manager import add_relation
        from src.models.relation import Relation

        py = create_entity("Python", "technology", "alpha",
                           observations=["Dynamic language", "Created by Guido"])
        proj = create_entity("memory-index", "project", "alpha",
                             observations=["MCP server for memory"])
        add_relation(Relation(
            id="rel1", from_entity=proj.id, to_entity=py.id,
            relation_type="uses", weight=1.0,
        ))
        return py, proj

    # ----- tests -----

    def test_export_creates_zip_with_expected_contents(self):
        from src.tools.portability import tool_export_vault

        self._seed_alpha()
        out_dir = Path(self.tmpdir) / "exports"
        result = tool_export_vault("alpha", str(out_dir))

        self.assertIn("Exported vault 'alpha'", result)
        zips = list(out_dir.glob("alpha_*.zip"))
        self.assertEqual(len(zips), 1)

        with zipfile.ZipFile(zips[0]) as zf:
            names = set(zf.namelist())
            self.assertEqual(names, {
                "manifest.json", "entities.json", "observations.json", "relations.json",
            })
            manifest = json.loads(zf.read("manifest.json"))
            self.assertEqual(manifest["source_vault"], "alpha")
            self.assertEqual(manifest["counts"]["entities"], 2)
            self.assertEqual(manifest["counts"]["observations"], 3)
            self.assertEqual(manifest["counts"]["relations"], 1)

    def test_export_unknown_vault(self):
        from src.tools.portability import tool_export_vault
        result = tool_export_vault("nope")
        self.assertIn("not found", result)

    def test_roundtrip_into_empty_vault(self):
        from src.tools.portability import tool_export_vault, tool_import_vault
        from src.indexer.store import list_entities, get_observations, get_entity_by_name
        from src.graph.manager import get_all_relations

        self._seed_alpha()
        export_path = Path(self.tmpdir) / "alpha.zip"
        tool_export_vault("alpha", str(export_path))

        # Import into a new vault
        result = tool_import_vault(str(export_path), "beta")
        self.assertIn("2 created", result)
        self.assertIn("3 added", result)  # observations
        self.assertIn("1 added", result)  # relations

        beta_entities, _ = list_entities(vault="beta", limit=100)
        self.assertEqual(len(beta_entities), 2)
        names = {e.name for e in beta_entities}
        self.assertEqual(names, {"Python", "memory-index"})

        py = get_entity_by_name("Python", "beta")
        self.assertEqual(len(get_observations(py.id)), 2)

        # Relation rebuilt with remapped IDs
        rels = [r for r in get_all_relations()
                if r.relation_type == "uses"]
        beta_proj = get_entity_by_name("memory-index", "beta")
        self.assertTrue(any(
            r.from_entity == beta_proj.id and r.to_entity == py.id for r in rels
        ))

    def test_import_is_additive_no_duplicates(self):
        from src.tools.portability import tool_export_vault, tool_import_vault
        from src.indexer.store import (
            create_entity, add_observation, get_observations, get_entity_by_name,
            list_entities,
        )
        from src.graph.manager import get_all_relations

        # Source data in alpha
        self._seed_alpha()
        export_path = Path(self.tmpdir) / "alpha.zip"
        tool_export_vault("alpha", str(export_path))

        # Pre-populate beta with overlap: same Python entity, one shared and one unique observation
        import src.config as config_mod
        config_mod.VAULTS["beta"] = config_mod.VaultConfig(name="beta", collection_name="memory_beta")
        py_beta = create_entity("Python", "technology", "beta",
                                observations=["Dynamic language",  # overlaps
                                              "Has GIL"])           # unique to beta
        # Beta also has an unrelated entity that should be untouched
        create_entity("Rust", "technology", "beta",
                      observations=["Memory safe"])

        before_beta_count = len(list_entities(vault="beta", limit=100)[0])
        self.assertEqual(before_beta_count, 2)

        # First import
        result1 = tool_import_vault(str(export_path), "beta")

        # Python should be reused, memory-index created
        self.assertIn("1 created", result1)
        self.assertIn("1 reused", result1)

        # Python obs: had {"Dynamic language", "Has GIL"} (2)
        # Imported: {"Dynamic language" (dup), "Created by Guido" (new)}
        # Result: 3 observations
        py_after = get_entity_by_name("Python", "beta")
        contents = {o.content for o in get_observations(py_after.id)}
        self.assertEqual(contents, {"Dynamic language", "Has GIL", "Created by Guido"})

        # Rust untouched
        rust = get_entity_by_name("Rust", "beta")
        self.assertEqual(len(get_observations(rust.id)), 1)

        # Re-import is fully idempotent
        result2 = tool_import_vault(str(export_path), "beta")
        self.assertIn("2 reused", result2)
        self.assertIn("0 added", result2)  # observations
        self.assertNotIn("1 added", result2)  # relations

        # Still exactly 3 obs on Python and exactly one 'uses' relation
        self.assertEqual(len(get_observations(py_after.id)), 3)
        uses = [r for r in get_all_relations() if r.relation_type == "uses"]
        # alpha already had one + we imported once -> beta has 1 + alpha still has 1 = 2 total
        self.assertEqual(len(uses), 2)

    def test_import_auto_creates_target_vault(self):
        from src.tools.portability import tool_export_vault, tool_import_vault
        from src.config import VAULTS

        self._seed_alpha()
        export_path = Path(self.tmpdir) / "alpha.zip"
        tool_export_vault("alpha", str(export_path))

        self.assertNotIn("gamma", VAULTS)
        tool_import_vault(str(export_path), "gamma")
        self.assertIn("gamma", VAULTS)

    def test_import_defaults_target_to_source_vault(self):
        from src.tools.portability import tool_export_vault, tool_import_vault
        from src.indexer.store import get_observations, get_entity_by_name

        self._seed_alpha()
        export_path = Path(self.tmpdir) / "alpha.zip"
        tool_export_vault("alpha", str(export_path))

        # Import without specifying vault — should target 'alpha' (the source)
        # Since the source vault still has the data, everything should dedupe.
        result = tool_import_vault(str(export_path))
        self.assertIn("vault 'alpha'", result)
        self.assertIn("2 reused", result)
        self.assertIn("0 added", result)

        py = get_entity_by_name("Python", "alpha")
        self.assertEqual(len(get_observations(py.id)), 2)

    def test_import_bad_archive(self):
        from src.tools.portability import tool_import_vault

        bad = Path(self.tmpdir) / "bad.zip"
        bad.write_bytes(b"not a zip")
        result = tool_import_vault(str(bad))
        self.assertIn("Error", result)

    def test_import_missing_file(self):
        from src.tools.portability import tool_import_vault
        result = tool_import_vault(str(Path(self.tmpdir) / "nope.zip"))
        self.assertIn("not found", result)

    # ----- history preservation -----

    def _seed_alpha_with_history(self):
        """An entity whose fact was revised twice, leaving a supersede chain."""
        from src.indexer.store import create_entity, add_observation

        ent = create_entity("Perception", "project", "alpha")
        v1 = add_observation(ent.id, "Uses .NET Framework")
        v2 = add_observation(ent.id, "Migrated to .NET 8", supersedes=v1.id)
        v3 = add_observation(ent.id, "Migrated to .NET 12", supersedes=v2.id)
        return ent, v1, v2, v3

    def test_export_includes_superseded_observations(self):
        from src.tools.portability import tool_export_vault

        self._seed_alpha_with_history()
        export_path = Path(self.tmpdir) / "alpha.zip"
        tool_export_vault("alpha", str(export_path))

        with zipfile.ZipFile(export_path) as zf:
            obs = json.loads(zf.read("observations.json"))
        self.assertEqual(len(obs), 3)
        self.assertEqual(sum(1 for o in obs if o.get("superseded_by")), 2)

    def test_import_preserves_superseded_history(self):
        """Import must not discard superseded rows — export->import was lossy."""
        from src.tools.portability import tool_export_vault, tool_import_vault
        from src.indexer.store import get_entity_by_name, get_observations

        self._seed_alpha_with_history()
        export_path = Path(self.tmpdir) / "alpha.zip"
        tool_export_vault("alpha", str(export_path))

        tool_import_vault(str(export_path), "beta")

        ent = get_entity_by_name("Perception", "beta")
        self.assertIsNotNone(ent)

        # All three rows made it across...
        all_obs = get_observations(ent.id, include_superseded=True)
        self.assertEqual(len(all_obs), 3)
        self.assertEqual(
            {o.content for o in all_obs},
            {"Uses .NET Framework", "Migrated to .NET 8", "Migrated to .NET 12"},
        )

        # ...but only the head of the chain is an ACTIVE fact.
        active = get_observations(ent.id)
        self.assertEqual([o.content for o in active], ["Migrated to .NET 12"])

    def test_import_remaps_superseded_by_pointers(self):
        """superseded_by must point at the NEW local IDs, not the archive's."""
        from src.tools.portability import tool_export_vault, tool_import_vault
        from src.indexer.store import get_entity_by_name, get_observations

        _, v1, v2, v3 = self._seed_alpha_with_history()
        old_ids = {v1.id, v2.id, v3.id}

        export_path = Path(self.tmpdir) / "alpha.zip"
        tool_export_vault("alpha", str(export_path))
        tool_import_vault(str(export_path), "beta")

        ent = get_entity_by_name("Perception", "beta")
        by_content = {o.content: o
                      for o in get_observations(ent.id, include_superseded=True)}
        local_ids = {o.id for o in by_content.values()}

        # Fresh IDs, and every pointer resolves within the new vault.
        self.assertEqual(local_ids & old_ids, set())
        self.assertEqual(
            by_content["Uses .NET Framework"].superseded_by,
            by_content["Migrated to .NET 8"].id,
        )
        self.assertEqual(
            by_content["Migrated to .NET 8"].superseded_by,
            by_content["Migrated to .NET 12"].id,
        )
        self.assertEqual(by_content["Migrated to .NET 12"].superseded_by, "")

    def test_import_superseded_never_lands_as_active(self):
        """Even when the replacement is missing from the archive, the old row
        must import as superseded — never resurface as a current fact."""
        from src.tools.portability import tool_import_vault
        from src.indexer.store import get_entity_by_name, get_observations

        archive = Path(self.tmpdir) / "handmade.zip"
        entities = [{
            "id": "E1", "name": "Perception", "entity_type": "project",
            "vault": "alpha", "deleted": False,
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:00+00:00",
        }]
        observations = [
            {"id": "O1", "entity_id": "E1", "content": "Old truth",
             "source": "", "created_at": "2026-01-01T00:00:00+00:00",
             "deleted": False, "superseded_by": "GONE"},
            {"id": "O2", "entity_id": "E1", "content": "Current truth",
             "source": "", "created_at": "2026-01-02T00:00:00+00:00",
             "deleted": False},
        ]
        manifest = {
            "format_version": 1, "source_vault": "alpha",
            "exported_at": "2026-01-03T00:00:00+00:00",
            "counts": {"entities": 1, "observations": 2, "relations": 0},
        }
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("manifest.json", json.dumps(manifest))
            zf.writestr("entities.json", json.dumps(entities))
            zf.writestr("observations.json", json.dumps(observations))
            zf.writestr("relations.json", json.dumps([]))

        tool_import_vault(str(archive), "beta")

        ent = get_entity_by_name("Perception", "beta")
        active = get_observations(ent.id)
        self.assertEqual([o.content for o in active], ["Current truth"])
        self.assertEqual(len(get_observations(ent.id, include_superseded=True)), 2)

    def _write_archive(self, name, entities, observations, relations=()):
        archive = Path(self.tmpdir) / name
        manifest = {
            "format_version": 1, "source_vault": "alpha",
            "exported_at": "2026-01-03T00:00:00+00:00",
            "counts": {"entities": len(entities),
                       "observations": len(observations),
                       "relations": len(relations)},
        }
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("manifest.json", json.dumps(manifest))
            zf.writestr("entities.json", json.dumps(entities))
            zf.writestr("observations.json", json.dumps(observations))
            zf.writestr("relations.json", json.dumps(list(relations)))
        return archive

    def test_import_never_supersedes_a_preexisting_active_observation(self):
        """Import is ADDITIVE — an archive's superseded row must not retire a
        fact the target vault already holds as current."""
        from src.tools.portability import tool_import_vault
        from src.indexer.store import (
            create_entity, get_entity_by_name, get_observations,
        )
        import src.config as config_mod

        config_mod.VAULTS["beta"] = config_mod.VaultConfig(
            name="beta", collection_name="memory_beta")
        create_entity("Steven", "person", "beta",
                      observations=["Steven works at Augmentus"])

        archive = self._write_archive(
            "history.zip",
            [{"id": "E1", "name": "Steven", "entity_type": "person",
              "vault": "alpha", "deleted": False,
              "created_at": "2026-01-01T00:00:00+00:00",
              "updated_at": "2026-01-01T00:00:00+00:00"}],
            [
                {"id": "OLD1", "entity_id": "E1",
                 "content": "Steven works at Augmentus", "source": "",
                 "created_at": "2026-01-01T00:00:00+00:00",
                 "deleted": False, "superseded_by": "OLD2"},
                {"id": "OLD2", "entity_id": "E1",
                 "content": "Steven works at Anthropic", "source": "",
                 "created_at": "2026-02-01T00:00:00+00:00", "deleted": False},
            ],
        )

        tool_import_vault(str(archive), "beta")

        ent = get_entity_by_name("Steven", "beta")
        active = {o.content for o in get_observations(ent.id)}
        # The pre-existing fact is untouched; the imported one is added.
        self.assertIn("Steven works at Augmentus", active)
        self.assertIn("Steven works at Anthropic", active)

    def test_import_keeps_row_active_when_an_active_archive_row_shares_it(self):
        """Two archive rows with identical content — one active, one
        superseded — dedupe onto one local row, which must stay active."""
        from src.tools.portability import tool_import_vault
        from src.indexer.store import get_entity_by_name, get_observations

        archive = self._write_archive(
            "dupes.zip",
            [{"id": "E1", "name": "Thing", "entity_type": "concept",
              "vault": "alpha", "deleted": False,
              "created_at": "2026-01-01T00:00:00+00:00",
              "updated_at": "2026-01-01T00:00:00+00:00"}],
            [
                # Superseded row first, so it is the one actually created.
                {"id": "A", "entity_id": "E1", "content": "Same fact",
                 "source": "", "created_at": "2026-01-01T00:00:00+00:00",
                 "deleted": False, "superseded_by": "C"},
                {"id": "B", "entity_id": "E1", "content": "Same fact",
                 "source": "", "created_at": "2026-01-02T00:00:00+00:00",
                 "deleted": False},
                {"id": "C", "entity_id": "E1", "content": "Other fact",
                 "source": "", "created_at": "2026-01-03T00:00:00+00:00",
                 "deleted": False},
            ],
        )

        tool_import_vault(str(archive), "beta")

        ent = get_entity_by_name("Thing", "beta")
        active = {o.content for o in get_observations(ent.id)}
        self.assertEqual(active, {"Same fact", "Other fact"})

    def test_import_skips_soft_deleted_observations(self):
        """Deleted rows are still dropped — only superseded ones now survive."""
        from src.tools.portability import tool_export_vault, tool_import_vault
        from src.indexer.store import (
            create_entity, add_observation, delete_observation,
            get_entity_by_name, get_observations,
        )

        ent = create_entity("Thing", "concept", "alpha")
        add_observation(ent.id, "Kept")
        gone = add_observation(ent.id, "Removed")
        delete_observation(gone.id)

        export_path = Path(self.tmpdir) / "alpha.zip"
        tool_export_vault("alpha", str(export_path))
        result = tool_import_vault(str(export_path), "beta")
        self.assertIn("skipped (deleted)", result)

        beta = get_entity_by_name("Thing", "beta")
        contents = {o.content
                    for o in get_observations(beta.id, include_superseded=True)}
        self.assertEqual(contents, {"Kept"})

    def test_history_roundtrip_is_idempotent(self):
        """Re-importing the same archive must not duplicate history."""
        from src.tools.portability import tool_export_vault, tool_import_vault
        from src.indexer.store import get_entity_by_name, get_observations

        self._seed_alpha_with_history()
        export_path = Path(self.tmpdir) / "alpha.zip"
        tool_export_vault("alpha", str(export_path))

        tool_import_vault(str(export_path), "beta")
        result2 = tool_import_vault(str(export_path), "beta")
        self.assertIn("0 added", result2)

        ent = get_entity_by_name("Perception", "beta")
        self.assertEqual(len(get_observations(ent.id, include_superseded=True)), 3)
        self.assertEqual(len(get_observations(ent.id)), 1)

    # ----- occurred_at -----

    def test_occurred_at_roundtrips(self):
        from src.tools.portability import tool_export_vault, tool_import_vault
        from src.indexer.store import (
            create_entity, add_observation, get_entity_by_name, get_observations,
        )

        ent = create_entity("Python", "technology", "alpha")
        add_observation(ent.id, "Created by Guido", occurred_at="1991-02-20")
        add_observation(ent.id, "No known event time")

        export_path = Path(self.tmpdir) / "alpha.zip"
        tool_export_vault("alpha", str(export_path))

        with zipfile.ZipFile(export_path) as zf:
            raw = json.loads(zf.read("observations.json"))
        by_content = {o["content"]: o for o in raw}
        self.assertEqual(by_content["Created by Guido"]["occurred_at"], "1991-02-20")
        # Unset stays absent from the archive, not serialized as null.
        self.assertNotIn("occurred_at", by_content["No known event time"])

        tool_import_vault(str(export_path), "beta")

        beta = get_entity_by_name("Python", "beta")
        imported = {o.content: o for o in get_observations(beta.id)}
        self.assertEqual(imported["Created by Guido"].occurred_at, "1991-02-20")
        self.assertIsNone(imported["No known event time"].occurred_at)

    # ----- concurrency -----

    def test_export_scan_survives_a_write_landing_mid_scan(self):
        """_collect_vault_data must read a snapshot, not the live dicts.

        Deterministic stand-in for the real race: a write is injected from
        inside the scan itself. Iterating the live dict raises
        "dictionary changed size during iteration"; iterating a snapshot
        taken under STORE_LOCK cannot.
        """
        from src.tools.portability import _collect_vault_data
        from src.indexer.store import create_entity
        from src.models.observation import Observation
        import src.indexer.store as store_mod

        ent = create_entity("Python", "technology", "alpha",
                            observations=[f"fact {i}" for i in range(20)])

        fired = []
        victim = next(iter(store_mod._observations.values()))
        original_to_dict = victim.to_dict

        def to_dict_that_writes():
            if not fired:
                fired.append(True)
                # Simulate a concurrent add_observation landing mid-scan.
                for i in range(5):
                    extra = Observation(id=f"injected{i}", entity_id=ent.id,
                                        content=f"injected {i}")
                    store_mod._observations[extra.id] = extra
            return original_to_dict()

        victim.to_dict = to_dict_that_writes

        data = _collect_vault_data("alpha")  # must not raise

        self.assertTrue(fired)
        self.assertGreaterEqual(len(data["observations"]), 20)

    def test_pipe_in_content_survives_roundtrip(self):
        """Contents are never delimiter-split anywhere in the pipeline."""
        from src.tools.portability import tool_export_vault, tool_import_vault
        from src.indexer.store import (
            create_entity, add_observation, get_entity_by_name, get_observations,
        )

        fact = "Runs `cat x | grep y | wc -l` in the harness"
        ent = create_entity("Shell", "concept", "alpha")
        add_observation(ent.id, fact)

        export_path = Path(self.tmpdir) / "alpha.zip"
        tool_export_vault("alpha", str(export_path))
        tool_import_vault(str(export_path), "beta")

        beta = get_entity_by_name("Shell", "beta")
        obs = get_observations(beta.id)
        self.assertEqual([o.content for o in obs], [fact])


if __name__ == "__main__":
    unittest.main()
