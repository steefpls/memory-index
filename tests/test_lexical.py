"""Tests for the BM25 index behind search's lexical candidates."""

import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.entity import Entity  # noqa: E402
from src.models.observation import Observation  # noqa: E402


class TestLexical(unittest.TestCase):
    def setUp(self):
        self.ents = {"e1": Entity(id="e1", name="Fleet backup", entity_type="project", vault="work"),
                     "e2": Entity(id="e2", name="Printer", entity_type="thing", vault="work")}
        self.obs = {"o1": Observation(id="o1", entity_id="e1", content="restore drill runs monthly"),
                    "o2": Observation(id="o2", entity_id="e2", content="the restore button is broken"),
                    "o3": Observation(id="o3", entity_id="e2", content="Brother laser, bought 2025")}
        p = patch("src.indexer.store.snapshot_store", side_effect=lambda: (dict(self.ents), dict(self.obs)))
        p.start()
        self.addCleanup(p.stop)

    def search(self, query, allowed=lambda oid: True, limit=10):
        from src.tools import lexical
        return lexical.search(query, allowed, limit)

    def test_entity_name_counts(self):
        self.assertEqual(self.search("fleet backup restore")[0], "o1")

    def test_no_shared_word_is_empty(self):
        self.assertEqual(self.search("favourite food"), [])

    def test_stopwords_alone_match_nothing(self):
        self.assertEqual(self.search("what is the"), [])

    def test_filter_and_limit(self):
        self.assertEqual(self.search("restore", allowed=lambda oid: oid != "o1"), ["o2"])
        self.assertEqual(len(self.search("restore", limit=1)), 1)

    def test_a_write_rebuilds_the_index(self):
        self.assertEqual(self.search("toner"), [])
        self.obs["o4"] = Observation(id="o4", entity_id="e2", content="toner ordered")
        self.assertEqual(self.search("toner"), ["o4"])

    def test_deleted_rows_are_left_out(self):
        self.obs["o2"].deleted = True
        self.assertEqual(self.search("restore"), ["o1"])


if __name__ == "__main__":
    unittest.main()
