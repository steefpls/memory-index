"""Tests for the Librarian — DBSCAN clustering and structural gap detection."""

import json
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestDBSCAN(unittest.TestCase):
    """Test DBSCAN clustering on embeddings."""

    def test_finds_clusters(self):
        from src.tools.librarian import _run_dbscan

        # Two tight clusters + one noise point
        cluster_a = np.random.randn(5, 10) * 0.1 + np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        cluster_b = np.random.randn(5, 10) * 0.1 + np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
        noise = np.random.randn(1, 10) * 5

        embeddings = np.vstack([cluster_a, cluster_b, noise]).astype(np.float32)
        labels = _run_dbscan(embeddings, eps=0.5, min_samples=2)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        self.assertGreaterEqual(n_clusters, 1)
        # Noise point should be labeled -1
        self.assertEqual(labels[-1], -1)

    def test_all_noise_with_tight_eps(self):
        from src.tools.librarian import _run_dbscan

        # Random scattered points with very tight eps
        embeddings = np.random.randn(10, 10).astype(np.float32) * 10
        labels = _run_dbscan(embeddings, eps=0.01, min_samples=2)

        # Everything should be noise
        self.assertTrue(all(l == -1 for l in labels))

    def test_single_cluster_with_loose_eps(self):
        from src.tools.librarian import _run_dbscan

        # Tight cluster
        embeddings = np.random.randn(10, 10).astype(np.float32) * 0.01
        labels = _run_dbscan(embeddings, eps=1.0, min_samples=2)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        self.assertEqual(n_clusters, 1)


class TestBuildClusters(unittest.TestCase):
    """Test cluster metadata building."""

    def test_builds_cluster_info(self):
        from src.tools.librarian import _build_clusters
        from src.indexer import store
        from src.models.entity import Entity

        # Mock store
        store._loaded = True
        store._entities = {
            "e1": Entity(id="e1", name="Alpha", entity_type="concept", vault="test"),
            "e2": Entity(id="e2", name="Beta", entity_type="concept", vault="test"),
        }
        store._observations = {}

        labels = np.array([0, 0, 0, 1, 1, -1])
        obs_ids = ["o1", "o2", "o3", "o4", "o5", "o6"]
        entity_ids = ["e1", "e1", "e1", "e2", "e2", "e1"]
        embeddings = np.random.randn(6, 10).astype(np.float32) * 0.1

        clusters = _build_clusters(labels, obs_ids, entity_ids, embeddings)

        self.assertEqual(len(clusters), 2)
        # Cluster 0 should have 3 observations
        self.assertEqual(clusters[0]["size"], 3)
        # Cluster 1 should have 2 observations
        self.assertEqual(clusters[1]["size"], 2)
        # Each cluster should have entity info
        self.assertTrue(any(e["entity_name"] == "Alpha" for e in clusters[0]["entities"]))
        self.assertTrue(any(e["entity_name"] == "Beta" for e in clusters[1]["entities"]))

        # Cleanup
        store._entities = {}
        store._loaded = False


class TestGapDetection(unittest.TestCase):
    """Test structural gap detection between clusters."""

    def test_detects_gap_when_no_edges(self):
        from src.tools.librarian import _detect_gaps
        import networkx as nx

        # Two clusters with similar centroids but no graph edges
        centroid_a = np.array([1.0, 0.0, 0.0])
        centroid_b = np.array([0.9, 0.1, 0.0])  # very similar

        clusters = [
            {
                "cluster_id": 0, "size": 5, "cohesion": 0.1,
                "entities": [{"entity_id": "e1", "entity_name": "A", "entity_type": "concept"}],
                "centroid": centroid_a,
            },
            {
                "cluster_id": 1, "size": 5, "cohesion": 0.1,
                "entities": [{"entity_id": "e2", "entity_name": "B", "entity_type": "concept"}],
                "centroid": centroid_b,
            },
        ]

        graph = nx.MultiDiGraph()
        graph.add_node("e1")
        graph.add_node("e2")
        # No edges between them

        gaps = _detect_gaps(clusters, graph)

        self.assertTrue(len(gaps) > 0)
        self.assertEqual(gaps[0]["cluster_a"], 0)
        self.assertEqual(gaps[0]["cluster_b"], 1)
        self.assertGreater(gaps[0]["similarity"], 0.6)

    def test_no_gap_when_well_connected(self):
        from src.tools.librarian import _detect_gaps
        import networkx as nx

        centroid_a = np.array([1.0, 0.0, 0.0])
        centroid_b = np.array([0.9, 0.1, 0.0])

        clusters = [
            {
                "cluster_id": 0, "size": 5, "cohesion": 0.1,
                "entities": [{"entity_id": "e1", "entity_name": "A", "entity_type": "concept"}],
                "centroid": centroid_a,
            },
            {
                "cluster_id": 1, "size": 5, "cohesion": 0.1,
                "entities": [{"entity_id": "e2", "entity_name": "B", "entity_type": "concept"}],
                "centroid": centroid_b,
            },
        ]

        graph = nx.MultiDiGraph()
        graph.add_edge("e1", "e2", key="r1", relation_type="related_to")
        graph.add_edge("e2", "e1", key="r2", relation_type="related_to")

        gaps = _detect_gaps(clusters, graph)
        self.assertEqual(len(gaps), 0)

    def test_no_gap_when_dissimilar(self):
        from src.tools.librarian import _detect_gaps
        import networkx as nx

        centroid_a = np.array([1.0, 0.0, 0.0])
        centroid_b = np.array([0.0, 0.0, 1.0])  # orthogonal

        clusters = [
            {
                "cluster_id": 0, "size": 5, "cohesion": 0.1,
                "entities": [{"entity_id": "e1", "entity_name": "A", "entity_type": "concept"}],
                "centroid": centroid_a,
            },
            {
                "cluster_id": 1, "size": 5, "cohesion": 0.1,
                "entities": [{"entity_id": "e2", "entity_name": "B", "entity_type": "concept"}],
                "centroid": centroid_b,
            },
        ]

        graph = nx.MultiDiGraph()
        gaps = _detect_gaps(clusters, graph)
        self.assertEqual(len(gaps), 0)


class TestReportFormatting(unittest.TestCase):
    """Test text report formatting."""

    def test_format_text(self):
        from src.tools.librarian import _format_report_text

        report = {
            "vault": "test",
            "total_observations": 20,
            "clusters_found": 2,
            "noise_observations": 3,
            "eps": 0.5,
            "min_samples": 2,
            "clusters": [
                {
                    "cluster_id": 0, "size": 10, "cohesion": 0.15,
                    "entities": [
                        {"entity_id": "e1", "entity_name": "Python", "entity_type": "technology"},
                        {"entity_id": "e2", "entity_name": "JavaScript", "entity_type": "technology"},
                    ],
                },
                {
                    "cluster_id": 1, "size": 7, "cohesion": 0.22,
                    "entities": [
                        {"entity_id": "e3", "entity_name": "Alice", "entity_type": "person"},
                    ],
                },
            ],
            "structural_gaps": [
                {
                    "cluster_a": 0, "cluster_b": 1,
                    "similarity": 0.72, "cross_edges": 0,
                    "suggest_from": ["Python"], "suggest_to": ["Alice"],
                },
            ],
        }

        text = _format_report_text(report)
        self.assertIn("Librarian Report", text)
        self.assertIn("2 clusters", text)
        self.assertIn("Python", text)
        self.assertIn("Structural Gaps", text)
        self.assertIn("Consider linking", text)

    def test_format_no_gaps(self):
        from src.tools.librarian import _format_report_text

        report = {
            "vault": "test",
            "total_observations": 10,
            "clusters_found": 1,
            "noise_observations": 0,
            "eps": 0.5,
            "min_samples": 2,
            "clusters": [{"cluster_id": 0, "size": 10, "cohesion": 0.1, "entities": []}],
            "structural_gaps": [],
        }

        text = _format_report_text(report)
        self.assertIn("none found", text)


class TestToolValidation(unittest.TestCase):
    """Test tool input validation."""

    def test_requires_vault(self):
        from src.tools.librarian import tool_run_librarian
        result = tool_run_librarian(vault="")
        self.assertIn("Error", result)
        self.assertIn("vault is required", result)

    def test_unknown_vault(self):
        from src.tools.librarian import tool_run_librarian
        result = tool_run_librarian(vault="nonexistent_vault_xyz")
        self.assertIn("Error", result)
        self.assertIn("Unknown vault", result)

    def test_invalid_format(self):
        from src.tools.librarian import tool_run_librarian
        result = tool_run_librarian(vault="work", output_format="xml")
        self.assertIn("Error", result)


if __name__ == "__main__":
    unittest.main()
