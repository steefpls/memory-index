"""Tests for graph analysis: spreading activation, PageRank, communities, knowledge gaps."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import networkx as nx


class TestSpreadingActivation(unittest.TestCase):
    """Test spreading activation algorithm."""

    def _setup_graph(self):
        """Create a test graph: A -> B -> C -> D, A -> E."""
        from src.graph import manager
        manager._graph = nx.MultiDiGraph()
        manager._relations = {}
        g = manager._graph
        g.add_edge("A", "B", key="r1", relation_type="related_to", weight=1.0)
        g.add_edge("B", "C", key="r2", relation_type="related_to", weight=1.0)
        g.add_edge("C", "D", key="r3", relation_type="related_to", weight=0.5)
        g.add_edge("A", "E", key="r4", relation_type="related_to", weight=1.0)
        return g

    def tearDown(self):
        from src.graph import manager
        manager._graph = None
        manager._relations = {}

    def test_basic_activation(self):
        self._setup_graph()
        from src.graph.traversal import spread_activation

        result = spread_activation({"A"}, decay=0.7, max_hops=2, top_k=10)

        # B and E should have highest activation (direct neighbors)
        self.assertIn("B", result)
        self.assertIn("E", result)
        # Seeds should NOT be in results
        self.assertNotIn("A", result)

    def test_decay_reduces_energy(self):
        self._setup_graph()
        from src.graph.traversal import spread_activation

        result = spread_activation({"A"}, decay=0.7, max_hops=3, top_k=10)

        # B (1 hop) should have more energy than C (2 hops)
        if "B" in result and "C" in result:
            self.assertGreater(result["B"], result["C"])

    def test_weight_affects_propagation(self):
        self._setup_graph()
        from src.graph.traversal import spread_activation

        result = spread_activation({"C"}, decay=0.7, max_hops=1, top_k=10)

        # B has weight 1.0 edge to C, D has weight 0.5 edge from C
        if "B" in result and "D" in result:
            self.assertGreater(result["B"], result["D"])

    def test_empty_seeds(self):
        self._setup_graph()
        from src.graph.traversal import spread_activation

        result = spread_activation(set(), decay=0.7, max_hops=2)
        self.assertEqual(result, {})

    def test_nonexistent_seed(self):
        self._setup_graph()
        from src.graph.traversal import spread_activation

        result = spread_activation({"NONEXISTENT"}, decay=0.7, max_hops=2)
        self.assertEqual(result, {})

    def test_lateral_inhibition(self):
        """With top_k=1, only the most activated neighbor should survive each hop."""
        self._setup_graph()
        from src.graph.traversal import spread_activation

        result = spread_activation({"A"}, decay=0.7, max_hops=1, top_k=1)
        # Only 1 neighbor should be in results (the one with highest energy)
        self.assertLessEqual(len(result), 1)

    def test_multiple_seeds(self):
        self._setup_graph()
        from src.graph.traversal import spread_activation

        result = spread_activation({"A", "D"}, decay=0.7, max_hops=2, top_k=10)

        # C should receive energy from both directions
        self.assertIn("C", result)


class TestPageRank(unittest.TestCase):
    def _setup_graph(self):
        from src.graph import manager
        manager._graph = nx.MultiDiGraph()
        manager._relations = {}
        g = manager._graph
        # Star topology: everything points to hub
        g.add_edge("s1", "hub", key="r1", relation_type="related_to", weight=1.0)
        g.add_edge("s2", "hub", key="r2", relation_type="related_to", weight=1.0)
        g.add_edge("s3", "hub", key="r3", relation_type="related_to", weight=1.0)
        g.add_edge("hub", "s4", key="r4", relation_type="related_to", weight=1.0)
        return g

    def tearDown(self):
        from src.graph import manager
        manager._graph = None
        manager._relations = {}

    def test_hub_ranks_high(self):
        self._setup_graph()
        from src.graph.traversal import compute_pagerank

        results = compute_pagerank(top_n=5)

        self.assertTrue(len(results) > 0)
        # Hub has 3 incoming edges — should be in top 2
        # (s4 may rank higher since hub funnels all rank to it)
        top_ids = [r["entity_id"] for r in results[:2]]
        self.assertIn("hub", top_ids)

    def test_empty_graph(self):
        from src.graph import manager
        manager._graph = nx.MultiDiGraph()
        manager._relations = {}
        from src.graph.traversal import compute_pagerank

        results = compute_pagerank()
        self.assertEqual(results, [])

    def test_top_n_limit(self):
        self._setup_graph()
        from src.graph.traversal import compute_pagerank

        results = compute_pagerank(top_n=2)
        self.assertLessEqual(len(results), 2)


class TestCommunityDetection(unittest.TestCase):
    def _setup_graph(self):
        from src.graph import manager
        manager._graph = nx.MultiDiGraph()
        manager._relations = {}
        g = manager._graph
        # Two clusters connected by a bridge
        # Cluster 1: A-B-C fully connected
        g.add_edge("A", "B", key="r1", relation_type="related_to", weight=1.0)
        g.add_edge("B", "C", key="r2", relation_type="related_to", weight=1.0)
        g.add_edge("C", "A", key="r3", relation_type="related_to", weight=1.0)
        # Cluster 2: D-E-F fully connected
        g.add_edge("D", "E", key="r4", relation_type="related_to", weight=1.0)
        g.add_edge("E", "F", key="r5", relation_type="related_to", weight=1.0)
        g.add_edge("F", "D", key="r6", relation_type="related_to", weight=1.0)
        # Bridge
        g.add_edge("C", "D", key="r7", relation_type="related_to", weight=0.1)
        return g

    def tearDown(self):
        from src.graph import manager
        manager._graph = None
        manager._relations = {}

    def test_detects_communities(self):
        self._setup_graph()
        from src.graph.traversal import detect_communities

        communities = detect_communities()
        # Should detect at least 1 community (may merge due to small graph)
        self.assertTrue(len(communities) >= 1)
        # Total members should equal total nodes
        total = sum(len(c) for c in communities)
        self.assertEqual(total, 6)

    def test_empty_graph(self):
        from src.graph import manager
        manager._graph = nx.MultiDiGraph()
        manager._relations = {}
        from src.graph.traversal import detect_communities

        communities = detect_communities()
        self.assertEqual(communities, [])

    def test_sorted_by_size(self):
        self._setup_graph()
        from src.graph.traversal import detect_communities

        communities = detect_communities()
        if len(communities) > 1:
            for i in range(len(communities) - 1):
                self.assertGreaterEqual(len(communities[i]), len(communities[i+1]))


class TestKnowledgeGaps(unittest.TestCase):
    def _setup_graph_and_store(self):
        from src.graph import manager
        from src.indexer import store
        manager._graph = nx.MultiDiGraph()
        manager._relations = {}
        g = manager._graph
        g.add_edge("e1", "e2", key="r1", relation_type="related_to", weight=1.0)
        g.add_edge("e3", "e1", key="r2", relation_type="related_to", weight=1.0)

        # Setup entities and observations
        store._loaded = True
        from src.models.entity import Entity
        from src.models.observation import Observation
        store._entities = {
            "e1": Entity(id="e1", name="WellDocumented", entity_type="concept", vault="test"),
            "e2": Entity(id="e2", name="UnderDocumented", entity_type="concept", vault="test"),
            "e3": Entity(id="e3", name="NoObs", entity_type="concept", vault="test"),
        }
        store._observations = {
            "o1": Observation(id="o1", entity_id="e1", content="Fact 1"),
            "o2": Observation(id="o2", entity_id="e1", content="Fact 2"),
            "o3": Observation(id="o3", entity_id="e1", content="Fact 3"),
            "o4": Observation(id="o4", entity_id="e2", content="Only fact"),
        }

    def tearDown(self):
        from src.graph import manager
        from src.indexer import store
        manager._graph = None
        manager._relations = {}
        store._entities = {}
        store._observations = {}
        store._loaded = False

    def test_finds_gaps(self):
        self._setup_graph_and_store()
        from src.graph.traversal import find_knowledge_gaps

        gaps = find_knowledge_gaps(min_observations=2)

        gap_ids = [g["entity_id"] for g in gaps]
        # e2 (1 observation) and e3 (0 observations) are gaps
        self.assertIn("e2", gap_ids)
        self.assertIn("e3", gap_ids)
        # e1 (3 observations) is NOT a gap
        self.assertNotIn("e1", gap_ids)

    def test_sorted_by_pagerank(self):
        self._setup_graph_and_store()
        from src.graph.traversal import find_knowledge_gaps

        gaps = find_knowledge_gaps(min_observations=2)
        if len(gaps) > 1:
            for i in range(len(gaps) - 1):
                self.assertGreaterEqual(gaps[i]["pagerank"], gaps[i+1]["pagerank"])


if __name__ == "__main__":
    unittest.main()
