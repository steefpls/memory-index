"""Tests for the short forms list views show (src/tools/brief.py): long facts
cut with the id that fetches them, sources down to a date the fact lacks."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.tools.brief import CLIP_AT, CLIP_SLACK, clip, source_tag  # noqa: E402


class TestSourceTag(unittest.TestCase):
    def test_date_from_source(self):
        self.assertEqual(source_tag("Claude Code session 2026-10-09 (desktop)", "a fact"),
                         " (2026-10-09)")

    def test_nothing_when_fact_has_a_date(self):
        self.assertEqual(source_tag("chat 2026-10-09", "Shipped 2026-10-08."), "")

    def test_nothing_when_source_has_no_date(self):
        self.assertEqual(source_tag("file: C:/bids/Part 2.pdf", "a fact"), "")
        self.assertEqual(source_tag("", "a fact"), "")


class TestClip(unittest.TestCase):
    def test_short_fact_untouched(self):
        text = "x" * (CLIP_AT + CLIP_SLACK)
        self.assertEqual(clip(text, "o1"), text)

    def test_long_fact_cut_at_a_word(self):
        text = " ".join(f"w{i}" for i in range(200))
        out = clip(text, "o1")
        shown, marker = out.split(" … [+")
        self.assertLessEqual(len(shown), CLIP_AT)
        self.assertTrue(text.startswith(shown + " "))
        hidden = len(text.split()) - len(shown.split())
        self.assertEqual(marker, f"{hidden} words, obs=o1]")

    def test_keyword_match_is_kept_in_view(self):
        text = "filler " * 100 + "C:/keys/kubeconfig is here " + "tail " * 50
        out = clip(text, "o1", match="c:/keys/kubeconfig")
        self.assertIn("C:/keys/kubeconfig", out)
        self.assertTrue(out.startswith("… "))
        self.assertIn(", obs=o1]", out)

    def test_match_already_in_view_keeps_the_start(self):
        text = "C:/keys first " + "filler " * 100
        self.assertTrue(clip(text, "o1", match="C:/keys").startswith("C:/keys first"))


if __name__ == "__main__":
    unittest.main()
