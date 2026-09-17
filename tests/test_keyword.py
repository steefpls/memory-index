"""Tests for search_memory keyword fallback: exact-substring scan over SQLite
when vector results are weak or the query looks like an exact string."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

TEST_THRESHOLDS = {"HIGH": 0.6, "MEDIUM": 1.0, "LOW": 1.4}


class KeywordTestCase(unittest.TestCase):
    """Real SQLite store (mocked vectors) + mocked search backend.

    Writes go through the real store with a mocked embedder; search_memory's
    vector pass is faked with chosen distances while get_observation /
    get_entity / snapshot_store hit the REAL rows — so the keyword scan sees
    exactly what was written.
    """

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

        from src.tools import search as search_mod
        self.search = search_mod
        search_mod._calibration_cache["test"] = dict(TEST_THRESHOLDS)

    def tearDown(self):
        self.search._calibration_cache.clear()
        from tests.support import close_sqlite
        close_sqlite()
        for p in self.patches:
            p.stop()
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    # ----- helpers -----

    def _write(self, entity, content, **kwargs):
        from src.indexer.store import get_entity_by_name, add_observation
        ent = get_entity_by_name(entity, "test")
        return add_observation(ent.id, content, **kwargs)

    def _search(self, query, vector_rows, **kwargs):
        """Run search_memory with a faked vector pass.

        vector_rows: list of (observation_id, distance) the fake Chroma
        returns. Everything else (store join, keyword scan) is real.
        """
        import types

        class _FakeColl:
            def query(self, query_embeddings=None, n_results=10, where=None,
                      include=None):
                rows = list(vector_rows)[:n_results]
                return {"ids": [[r[0] for r in rows]],
                        "distances": [[r[1] for r in rows]]}

        kwargs.setdefault("vault", "test")
        kwargs.setdefault("output_format", "json")
        with patch.object(self.search, "VAULTS", {"test": object()}), \
             patch("src.tools.search.get_vault",
                   return_value=types.SimpleNamespace(collection_name="test")), \
             patch("src.tools.search.get_collection",
                   return_value=_FakeColl()), \
             patch("src.tools.search._get_query_embeddings_with_guard",
                   return_value=[[0.1] * 8]):
            return json.loads(self.search.search_memory(query, **kwargs))

    def _search_text(self, query, vector_rows, **kwargs):
        import types

        class _FakeColl:
            def query(self, query_embeddings=None, n_results=10, where=None,
                      include=None):
                rows = list(vector_rows)[:n_results]
                return {"ids": [[r[0] for r in rows]],
                        "distances": [[r[1] for r in rows]]}

        kwargs.setdefault("vault", "test")
        with patch.object(self.search, "VAULTS", {"test": object()}), \
             patch("src.tools.search.get_vault",
                   return_value=types.SimpleNamespace(collection_name="test")), \
             patch("src.tools.search.get_collection",
                   return_value=_FakeColl()), \
             patch("src.tools.search._get_query_embeddings_with_guard",
                   return_value=[[0.1] * 8]):
            return self.search.search_memory(query, **kwargs)


class TestKeywordFallback(KeywordTestCase):
    def test_file_path_query_finds_exact_match(self):
        from src.tools.entities import tool_create_entity
        tool_create_entity("Deploy", "concept", "test")
        obs = self._write("Deploy", "TLS key lives at C:\\keys\\prod.pem on steef-server")

        payload = self._search("prod.pem", vector_rows=[(obs.id, 1.9)])
        by_id = {r["observation_id"]: r for r in payload["results"]}
        self.assertIn(obs.id, by_id)
        hit = by_id[obs.id]
        # Upgraded in place: the real (weak) vector score is kept, and the
        # exact-substring property is labelled on top of it.
        self.assertTrue(hit["keyword_match"])
        self.assertEqual(hit["distance"], 1.9)

    def test_keyword_only_hit_when_vector_misses_entirely(self):
        """The realistic embedding failure: the vector top-k never contained
        the exact match, so it is appended with NO fabricated score."""
        from src.tools.entities import tool_create_entity
        tool_create_entity("Deploy", "concept", "test")
        tool_create_entity("Other", "concept", "test")
        self._write("Deploy", "TLS key lives at C:\\keys\\prod.pem on steef-server")
        noise = self._write("Other", "unrelated note about gardening")

        payload = self._search("prod.pem", vector_rows=[(noise.id, 1.9)])
        by_id = {r["observation_id"]: r for r in payload["results"]}
        kw = [r for r in payload["results"] if r["keyword_match"]]
        self.assertEqual(len(kw), 1)
        self.assertIn("prod.pem", kw[0]["content"])
        self.assertEqual(kw[0]["confidence"], "KEYWORD")
        self.assertIsNone(kw[0]["distance"])
        self.assertIsNone(kw[0]["relevance_pct"])

    def test_number_query_finds_exact_match(self):
        from src.tools.entities import tool_create_entity
        tool_create_entity("Steven", "person", "test")
        obs = self._write("Steven", "Augmentus salary is 5.8k")

        payload = self._search("5.8k", vector_rows=[(obs.id, 2.0)])
        by_id = {r["observation_id"]: r for r in payload["results"]}
        self.assertIn(obs.id, by_id)
        self.assertTrue(by_id[obs.id]["keyword_match"])

    def test_text_output_tags_keyword_hits(self):
        from src.tools.entities import tool_create_entity
        tool_create_entity("Deploy", "concept", "test")
        self._write("Deploy", "TLS key lives at C:\\keys\\prod.pem")

        text = self._search_text("prod.pem", vector_rows=[])
        self.assertIn("keyword", text)
        self.assertIn("prod.pem", text)
        # No fabricated score on the keyword line.
        for line in text.splitlines():
            if "prod.pem" in line and "keyword" in line:
                self.assertNotIn("%", line)
                break
        else:
            self.fail("expected a tagged keyword line for prod.pem")

    def test_weak_vector_triggers_keyword_for_plain_query(self):
        """No digits/dots/slashes in the query — the weak vector result alone
        must still trigger the substring scan."""
        from src.tools.entities import tool_create_entity
        tool_create_entity("Garden", "concept", "test")
        obs = self._write("Garden", "the quokka sanctuary opens at dawn")

        payload = self._search("quokka sanctuary", vector_rows=[(obs.id, 1.9)])
        by_id = {r["observation_id"]: r for r in payload["results"]}
        self.assertIn(obs.id, by_id)
        self.assertTrue(by_id[obs.id]["keyword_match"])

    def test_strong_vector_plus_exact_query_appends_keyword(self):
        """Vector search stays primary: strong hits keep their rank, the exact
        match is labelled where it stands — never ranked with a fake score."""
        from src.tools.entities import tool_create_entity
        tool_create_entity("Lang", "technology", "test")
        tool_create_entity("Deploy", "concept", "test")
        strong = self._write("Lang", "Python is a programming language")
        exact = self._write("Deploy", "deploy key file prod.pem rotated")

        payload = self._search(
            "prod.pem", vector_rows=[(strong.id, 0.1), (exact.id, 1.9)])
        ids = [r["observation_id"] for r in payload["results"]]
        # Strong vector hit first, untouched...
        self.assertEqual(ids[0], strong.id)
        self.assertFalse(payload["results"][0]["keyword_match"])
        # ...exact match labelled in place (real weak score kept).
        self.assertIn(exact.id, ids)
        by_id = {r["observation_id"]: r for r in payload["results"]}
        self.assertTrue(by_id[exact.id]["keyword_match"])
        self.assertEqual(payload["keyword_matches"], 1)

    def test_no_keyword_section_for_plain_strong_query(self):
        from src.tools.entities import tool_create_entity
        tool_create_entity("Lang", "technology", "test")
        obs = self._write("Lang", "Python is a programming language")

        payload = self._search(
            "programming language", vector_rows=[(obs.id, 0.1)])
        self.assertEqual(payload["keyword_matches"], 0)
        self.assertFalse(payload["results"][0]["keyword_match"])

    def test_keyword_upgrades_in_place_never_duplicates(self):
        """An observation already returned by the vector pass is labelled
        where it stands, not repeated as a second row."""
        from src.tools.entities import tool_create_entity
        tool_create_entity("Deploy", "concept", "test")
        obs = self._write("Deploy", "rotate prod.pem monthly")

        payload = self._search("prod.pem", vector_rows=[(obs.id, 0.1)])
        ids = [r["observation_id"] for r in payload["results"]]
        self.assertEqual(ids.count(obs.id), 1)
        by_id = {r["observation_id"]: r for r in payload["results"]}
        self.assertTrue(by_id[obs.id]["keyword_match"])
        self.assertEqual(payload["keyword_matches"], 1)

    def test_keyword_matches_entity_names(self):
        from src.tools.entities import tool_create_entity
        tool_create_entity("prod-pem-rotation", "process", "test")
        from src.indexer.store import get_entity_by_name, add_observation
        ent = get_entity_by_name("prod-pem-rotation", "test")
        obs = add_observation(ent.id, "quarterly chore")

        payload = self._search("prod-pem-rotation", vector_rows=[])
        by_id = {r["observation_id"]: r for r in payload["results"]}
        self.assertIn(obs.id, by_id)
        self.assertTrue(by_id[obs.id]["keyword_match"])


class TestKeywordFilters(KeywordTestCase):
    def test_superseded_hidden_by_default(self):
        from src.tools.entities import tool_create_entity
        from src.tools.search import _keyword_search
        tool_create_entity("Svc", "concept", "test")
        old = self._write("Svc", "token file old.pem in use")

        # Supersede it so the old row is history, not current.
        from src.indexer.store import get_entity_by_name, add_observation
        ent = get_entity_by_name("Svc", "test")
        add_observation(ent.id, "token file new.pem in use", supersedes=old.id)

        hidden = _keyword_search("old.pem", ["test"], "", False,
                                 (None, None, "record"), set(), 5)
        self.assertEqual(hidden, [])
        shown = _keyword_search("old.pem", ["test"], "", True,
                                (None, None, "record"), set(), 5)
        self.assertEqual(len(shown), 1)
        self.assertTrue(shown[0]["superseded"])

    def test_entity_type_filter_applies(self):
        from src.tools.entities import tool_create_entity
        from src.tools.search import _keyword_search
        tool_create_entity("Person1", "person", "test")
        tool_create_entity("Tool1", "technology", "test")
        self._write("Person1", "owns key prod.pem backup")
        self._write("Tool1", "reads key prod.pem backup")

        person_hits = _keyword_search("prod.pem", ["test"], "person", False,
                                      (None, None, "record"), set(), 5)
        self.assertEqual(len(person_hits), 1)
        self.assertEqual(person_hits[0]["entity_name"], "Person1")

    def test_looks_like_exact_query(self):
        from src.tools.search import _looks_like_exact_query
        self.assertTrue(_looks_like_exact_query("prod.pem"))
        self.assertTrue(_looks_like_exact_query("C:\\keys\\prod.pem"))
        self.assertTrue(_looks_like_exact_query("/etc/ssl/cert.pem"))
        self.assertTrue(_looks_like_exact_query("5.8k"))
        self.assertTrue(_looks_like_exact_query("my_variable"))
        self.assertTrue(_looks_like_exact_query("version 2"))
        self.assertFalse(_looks_like_exact_query("programming language"))
        self.assertFalse(_looks_like_exact_query("who works on memory"))


if __name__ == "__main__":
    unittest.main()
