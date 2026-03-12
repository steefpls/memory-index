"""Graph traversal utilities — neighbor queries and graph-boost scoring."""

import logging
from typing import Any

import networkx as nx

from src.graph.manager import get_graph, get_relations_for_entity
from src.models.relation import Relation

logger = logging.getLogger(__name__)


def get_neighbors(entity_id: str, max_depth: int = 1,
                  relation_type: str | None = None) -> list[dict]:
    """Get neighboring entities via graph traversal.

    Args:
        entity_id: Starting entity ID.
        max_depth: How many hops to traverse (default 1).
        relation_type: Optional filter by relation type.

    Returns:
        List of neighbor dicts with entity_id, relation_type, direction, depth.
    """
    graph = get_graph()
    if not graph.has_node(entity_id):
        return []

    neighbors = []
    visited = {entity_id}

    # BFS traversal
    current_level = [entity_id]
    for depth in range(1, max_depth + 1):
        next_level = []
        for node in current_level:
            # Outgoing edges
            for _, target, data in graph.out_edges(node, data=True):
                if target in visited:
                    continue
                rel_type = data.get("relation_type", "related_to")
                if relation_type and rel_type != relation_type:
                    continue
                visited.add(target)
                next_level.append(target)
                neighbors.append({
                    "entity_id": target,
                    "relation_type": rel_type,
                    "direction": "outgoing",
                    "depth": depth,
                    "weight": data.get("weight", 1.0),
                    "context": data.get("context", ""),
                })

            # Incoming edges
            for source, _, data in graph.in_edges(node, data=True):
                if source in visited:
                    continue
                rel_type = data.get("relation_type", "related_to")
                if relation_type and rel_type != relation_type:
                    continue
                visited.add(source)
                next_level.append(source)
                neighbors.append({
                    "entity_id": source,
                    "relation_type": rel_type,
                    "direction": "incoming",
                    "depth": depth,
                    "weight": data.get("weight", 1.0),
                    "context": data.get("context", ""),
                })

        current_level = next_level

    return neighbors


def get_graph_boost_entity_ids(entity_ids: set[str], max_depth: int = 1) -> dict[str, float]:
    """Get entity IDs that should receive a graph boost in search results.

    For each input entity, finds 1-hop neighbors and assigns a boost score
    based on relation weight and depth.

    Args:
        entity_ids: Set of entity IDs from vector search results.
        max_depth: Traversal depth (default 1).

    Returns:
        Dict mapping entity_id -> boost_score (0.0 to 1.0).
    """
    graph = get_graph()
    boosted: dict[str, float] = {}

    for eid in entity_ids:
        neighbors = get_neighbors(eid, max_depth=max_depth)
        for nb in neighbors:
            nb_id = nb["entity_id"]
            if nb_id in entity_ids:
                continue  # already in results, don't boost
            weight = nb.get("weight", 1.0)
            depth = nb.get("depth", 1)
            # Boost decays with depth: 0.3 at depth 1, 0.15 at depth 2
            boost = 0.3 * weight / depth
            boosted[nb_id] = max(boosted.get(nb_id, 0.0), boost)

    return boosted


def get_graph_summary() -> dict:
    """Get summary statistics about the knowledge graph."""
    graph = get_graph()
    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()

    # Relation type distribution
    type_counts: dict[str, int] = {}
    for _, _, data in graph.edges(data=True):
        rt = data.get("relation_type", "unknown")
        type_counts[rt] = type_counts.get(rt, 0) + 1

    # Connected components (treat as undirected for this)
    if n_nodes > 0:
        undirected = graph.to_undirected()
        n_components = nx.number_connected_components(undirected)
    else:
        n_components = 0

    return {
        "nodes": n_nodes,
        "edges": n_edges,
        "components": n_components,
        "relation_types": type_counts,
    }
