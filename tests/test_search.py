"""Tests for hybrid search scoring and formatting."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestSearchScoring(unittest.TestCase):
    """Test confidence labels and normalized scores."""

    def test_confidence_labels(self):
        from src.tools.search import _confidence_label, _calibration_cache

        # Set up known thresholds
        _calibration_cache["test"] = {"HIGH": 650, "MEDIUM": 775, "LOW": 875}

        self.assertEqual(_confidence_label(500, "test"), "HIGH")
        self.assertEqual(_confidence_label(700, "test"), "MEDIUM")
        self.assertEqual(_confidence_label(800, "test"), "LOW")
        self.assertEqual(_confidence_label(900, "test"), "NO MATCH")

        _calibration_cache.clear()

    def test_normalized_score(self):
        from src.tools.search import _normalized_score, _calibration_cache

        _calibration_cache["test"] = {"HIGH": 650, "MEDIUM": 775, "LOW": 875}

        # Perfect match
        self.assertEqual(_normalized_score(0, "test"), 100.0)

        # HIGH boundary -> 60%
        score_at_high = _normalized_score(650, "test")
        self.assertAlmostEqual(score_at_high, 60.0, places=0)

        # MEDIUM boundary -> 35%
        score_at_med = _normalized_score(775, "test")
        self.assertAlmostEqual(score_at_med, 35.0, places=0)

        # LOW boundary -> 15%
        score_at_low = _normalized_score(875, "test")
        self.assertAlmostEqual(score_at_low, 15.0, places=0)

        # Far away -> 0%
        score_beyond = _normalized_score(1200, "test")
        self.assertEqual(score_beyond, 0.0)

        _calibration_cache.clear()

    def test_normalized_score_monotonic(self):
        """Scores should decrease as distance increases."""
        from src.tools.search import _normalized_score, _calibration_cache

        _calibration_cache["test"] = {"HIGH": 650, "MEDIUM": 775, "LOW": 875}

        prev_score = 100.0
        for dist in range(0, 1100, 50):
            score = _normalized_score(dist, "test")
            self.assertLessEqual(score, prev_score + 0.1,
                                 f"Score increased at distance {dist}")
            prev_score = score

        _calibration_cache.clear()

    def test_format_text(self):
        from src.tools.search import _format_text, _calibration_cache

        _calibration_cache["test"] = {"HIGH": 650, "MEDIUM": 775, "LOW": 875}

        results = [{
            "entity_id": "e1",
            "entity_name": "Python",
            "entity_type": "technology",
            "vault": "test",
            "distance": 500,
            "graph_boosted": False,
        }]
        entity_obs = {
            "e1": [{"content": "General purpose language", "source": ""}]
        }

        text = _format_text(results, entity_obs, "test query")
        self.assertIn("Python", text)
        self.assertIn("technology", text)
        self.assertIn("General purpose language", text)

        _calibration_cache.clear()


if __name__ == "__main__":
    unittest.main()
