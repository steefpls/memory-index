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


def spread_activation(seed_ids: set[str], decay: float = 0.7,
                      max_hops: int = 3, top_k: int = 10) -> dict[str, float]:
    """Spreading activation from seed entities through the graph.

    Energy starts at 1.0 on each seed, then propagates to neighbors with
    decay per hop. Energy accumulates across paths. Lateral inhibition
    keeps only top-K per hop to prevent explosion.

    Args:
        seed_ids: Starting entity IDs (typically from vector search).
        decay: Energy decay per hop (0.0 to 1.0, default 0.7).
        max_hops: Max traversal depth (default 3).
        top_k: Keep only top-K activated nodes per hop (default 10).

    Returns:
        Dict mapping entity_id -> activation energy (higher = more relevant).
    """
    graph = get_graph()
    activation: dict[str, float] = {sid: 1.0 for sid in seed_ids
                                     if graph.has_node(sid)}

    if not activation:
        return {}

    # Track all accumulated energy
    total_energy: dict[str, float] = dict(activation)
    visited_seeds = set(seed_ids)

    frontier = dict(activation)
    for hop in range(max_hops):
        next_frontier: dict[str, float] = {}

        for node_id, energy in frontier.items():
            # Outgoing edges
            for _, target, data in graph.out_edges(node_id, data=True):
                weight = data.get("weight", 1.0)
                propagated = energy * decay * weight
                next_frontier[target] = next_frontier.get(target, 0.0) + propagated

            # Incoming edges
            for source, _, data in graph.in_edges(node_id, data=True):
                weight = data.get("weight", 1.0)
                propagated = energy * decay * weight
                next_frontier[source] = next_frontier.get(source, 0.0) + propagated

        if not next_frontier:
            break

        # Lateral inhibition: keep only top-K
        sorted_next = sorted(next_frontier.items(), key=lambda x: -x[1])[:top_k]
        frontier = dict(sorted_next)

        # Accumulate energy
        for node_id, energy in frontier.items():
            total_energy[node_id] = total_energy.get(node_id, 0.0) + energy

    # Remove seeds from results (caller already has them)
    for sid in seed_ids:
        total_energy.pop(sid, None)

    return total_energy


def compute_pagerank(top_n: int = 20, alpha: float = 0.85) -> list[dict]:
    """Compute PageRank over the knowledge graph.

    Args:
        top_n: Number of top-ranked entities to return (default 20).
        alpha: Damping factor (default 0.85).

    Returns:
        List of dicts with entity_id and pagerank score, sorted descending.
    """
    graph = get_graph()
    if graph.number_of_nodes() == 0:
        return []

    try:
        pr = nx.pagerank(graph, alpha=alpha)
    except nx.PowerIterationFailedConvergence:
        logger.warning("PageRank failed to converge, using default alpha=0.85")
        pr = nx.pagerank(graph, alpha=0.85, max_iter=200)

    sorted_pr = sorted(pr.items(), key=lambda x: -x[1])[:top_n]
    return [{"entity_id": eid, "pagerank": round(score, 6)}
            for eid, score in sorted_pr]


def detect_communities() -> list[list[str]]:
    """Detect communities in the knowledge graph using Louvain method.

    Returns:
        List of communities, each a list of entity IDs. Sorted by size descending.
    """
    graph = get_graph()
    if graph.number_of_nodes() == 0:
        return []

    undirected = graph.to_undirected()
    try:
        communities = nx.community.louvain_communities(undirected)
    except Exception as e:
        logger.warning("Louvain community detection failed: %s", e)
        return []

    # Sort by size descending, convert sets to lists
    sorted_communities = sorted(communities, key=len, reverse=True)
    return [list(c) for c in sorted_communities]


def find_knowledge_gaps(min_observations: int = 2) -> list[dict]:
    """Find entities that are important (high PageRank) but under-documented.

    An entity with high graph centrality but few observations is a knowledge
    gap — it's referenced a lot but we don't know much about it.

    Args:
        min_observations: Entities with fewer observations than this are
                          considered under-documented (default 2).

    Returns:
        List of dicts with entity_id, pagerank, observation_count.
    """
    from src.indexer.store import get_observations, get_entity

    graph = get_graph()
    if graph.number_of_nodes() == 0:
        return []

    pr = nx.pagerank(graph, alpha=0.85)

    gaps = []
    for eid, score in pr.items():
        ent = get_entity(eid)
        if ent is None:
            continue
        obs_count = len(get_observations(eid))
        if obs_count < min_observations:
            gaps.append({
                "entity_id": eid,
                "entity_name": ent.name,
                "entity_type": ent.entity_type,
                "vault": ent.vault,
                "pagerank": round(score, 6),
                "observation_count": obs_count,
            })

    gaps.sort(key=lambda x: -x["pagerank"])
    return gaps


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
