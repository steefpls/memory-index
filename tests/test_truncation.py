"""Tests for the long-text guard: over-limit input (tokenizer-measured, not
char-counted) returns a warning while the full text is still stored."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

WARN_PHRASE = "truncated for search, split into smaller atomic facts"
LONG_TEXT = ("lorem ipsum dolor sit amet " * 500).strip()  # no trailing space:
# the store strips observation text, so the fixture must already be stripped
# for exact-equality assertions on what SQLite holds.


class TruncationTestCase(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.patches = [
            patch("src.config.DATA_DIR", Path(self.tmpdir)),
            patch("src.config.DB_FILE", Path(self.tmpdir) / "memory.db"),
            patch("src.config.CHROMA_DIR", Path(self.tmpdir) / "chroma"),
        ]
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
        config_mod.VAULTS["test"] = config_mod.VaultConfig(
            name="test", collection_name="memory_test")

    def tearDown(self):
        from tests.support import close_sqlite
        close_sqlite()
        for p in self.patches:
            p.stop()
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)


class TestLongTextGuard(TruncationTestCase):
    def test_add_observation_warns_but_stores_full_text(self):
        from src.tools.entities import tool_create_entity, tool_add_observation
        from src.indexer.store import get_entity_by_name, get_observations

        tool_create_entity("Doc", "reference", "test")
        with patch("src.indexer.embedder.count_tokens",
                   return_value=[9999]) as mock_count:
            result = tool_add_observation("Doc", LONG_TEXT, vault="test")

        self.assertIn("Observation added", result)
        self.assertIn(WARN_PHRASE, result)
        # The guard measured the prefixed embed text, not the raw content.
        counted = mock_count.call_args[0][0][0]
        self.assertIn("reference: Doc", counted)
        self.assertIn(LONG_TEXT, counted)
        # Full text still stored verbatim in SQLite.
        ent = get_entity_by_name("Doc", "test")
        stored = get_observations(ent.id)
        self.assertEqual(len(stored), 1)
        self.assertEqual(stored[0].content, LONG_TEXT)

    def test_add_observation_silent_when_under_limit(self):
        from src.tools.entities import tool_create_entity, tool_add_observation

        tool_create_entity("Doc", "reference", "test")
        with patch("src.indexer.embedder.count_tokens", return_value=[42]):
            result = tool_add_observation("Doc", "a short fact", vault="test")

        self.assertIn("Observation added", result)
        self.assertNotIn("Warning", result)
        self.assertNotIn(WARN_PHRASE, result)

    def test_add_observations_batch_warns_with_over_count(self):
        from src.tools.entities import tool_create_entity, tool_add_observations
        from src.indexer.store import get_entity_by_name, get_observations

        tool_create_entity("Doc", "reference", "test")
        with patch("src.indexer.embedder.count_tokens",
                   return_value=[10, 9999, 9999]):
            result = tool_add_observations(
                "Doc", ["fine", LONG_TEXT, LONG_TEXT + "more"], vault="test")

        self.assertIn("Added 3 observations", result)
        self.assertIn(WARN_PHRASE, result)
        self.assertIn("2 observation(s)", result)
        ent = get_entity_by_name("Doc", "test")
        contents = {o.content for o in get_observations(ent.id)}
        self.assertIn(LONG_TEXT, contents)

    def test_create_entity_with_long_initial_observations_warns(self):
        from src.tools.entities import tool_create_entity
        from src.indexer.store import get_entity_by_name, get_observations

        with patch("src.indexer.embedder.count_tokens", return_value=[8888]):
            result = tool_create_entity("Doc", "reference", "test",
                                        observations=[LONG_TEXT])

        self.assertIn("Entity created", result)
        self.assertIn(WARN_PHRASE, result)
        ent = get_entity_by_name("Doc", "test")
        self.assertEqual(get_observations(ent.id)[0].content, LONG_TEXT)

    def test_tokenizer_unavailable_means_no_warning_no_crash(self):
        from src.tools.entities import tool_create_entity, tool_add_observation

        tool_create_entity("Doc", "reference", "test")
        with patch("src.indexer.embedder.count_tokens", return_value=None):
            result = tool_add_observation("Doc", LONG_TEXT, vault="test")

        self.assertIn("Observation added", result)
        self.assertNotIn(WARN_PHRASE, result)

    def test_real_tokenizer_measures_tokens_not_characters(self):
        """End-to-end guard check with the real tokenizer when the model is
        present; vacuously passes (skips) where it is not downloaded."""
        from src.indexer.embedder import count_tokens

        counts = count_tokens(["hello world"])
        if counts is None:
            self.skipTest("tokenizer model not downloaded")
        self.assertEqual(len(counts), 1)
        short, = counts
        # A long word-salad is many tokens...
        many = count_tokens(["lorem ipsum " * 2000])[0]
        self.assertGreater(many, short)
        # ...and crosses the configured window.
        from src.config import EMBED_MAX_TOKENS
        self.assertGreater(many, EMBED_MAX_TOKENS)
        self.assertLessEqual(short, EMBED_MAX_TOKENS)


if __name__ == "__main__":
    unittest.main()
