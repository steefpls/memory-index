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
            ("src.config.ENTITIES_FILE", tmp / "memory_entities.json"),
            ("src.config.VAULTS_FILE", tmp / "vaults.json"),
            ("src.config.GRAPH_FILE", tmp / "memory_graph.json"),
            ("src.config.CHROMA_DIR", tmp / "chroma"),
            ("src.indexer.store.DATA_DIR", tmp),
            ("src.indexer.store.ENTITIES_FILE", tmp / "memory_entities.json"),
            ("src.graph.manager.GRAPH_FILE", tmp / "memory_graph.json"),
            ("src.graph.manager.DATA_DIR", tmp),
            ("src.tools.portability.DATA_DIR", tmp),
        ]
        for target, value in path_patches:
            p = patch(target, value)
            self.patches.append(p)
            p.start()

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


if __name__ == "__main__":
    unittest.main()
