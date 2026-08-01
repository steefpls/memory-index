"""NetworkX MultiDiGraph wrapper with SQLite persistence.

Manages the relation graph between entities. In-memory for fast traversal,
persisted row-by-row to SQLite for durability.
"""

import logging
import threading

import networkx as nx

from src.indexer import db
from src.models.relation import Relation

logger = logging.getLogger(__name__)

_graph: nx.MultiDiGraph | None = None
_relations: dict[str, Relation] = {}  # keyed by relation ID

# Guards every mutation, save, and full-dict scan of _relations / _graph.
# Atomic file replacement alone is not enough: the PAYLOAD itself is torn if
# _relations is serialized while another thread mutates it. Re-entrant so
# nested calls (remove_entity_relations -> remove_relation -> _save_graph)
# don't self-deadlock.
GRAPH_LOCK = threading.RLock()


def _get_graph() -> nx.MultiDiGraph:
    """Get or create the singleton graph."""
    global _graph
    with GRAPH_LOCK:
        if _graph is None:
            _graph = nx.MultiDiGraph()
            _load_graph()
        return _graph


def _load_graph() -> None:
    """Load graph and relations from SQLite."""
    global _graph, _relations
    with GRAPH_LOCK:
        if _graph is None:
            _graph = nx.MultiDiGraph()

        try:
            for rd in db.load_relations():
                rel = Relation.from_dict(rd)
                _relations[rel.id] = rel
                _graph.add_edge(
                    rel.from_entity, rel.to_entity,
                    key=rel.id,
                    relation_type=rel.relation_type,
                    weight=rel.weight,
                    context=rel.context,
                    created_at=rel.created_at,
                )
            logger.info("Loaded graph: %d nodes, %d edges, %d relations",
                        _graph.number_of_nodes(), _graph.number_of_edges(),
                        len(_relations))
        except Exception as e:
            logger.warning("Failed to load graph: %s", e)


def add_relation(relation: Relation) -> None:
    """Add a relation to the graph."""
    with GRAPH_LOCK:
        graph = _get_graph()
        _relations[relation.id] = relation
        graph.add_edge(
            relation.from_entity, relation.to_entity,
            key=relation.id,
            relation_type=relation.relation_type,
            weight=relation.weight,
            context=relation.context,
            created_at=relation.created_at,
        )
        db.upsert_relations([relation])
    logger.info("Added relation: %s -[%s]-> %s",
                relation.from_entity, relation.relation_type, relation.to_entity)


def remove_relation(relation_id: str) -> bool:
    """Remove a relation from the graph."""
    with GRAPH_LOCK:
        graph = _get_graph()
        rel = _relations.pop(relation_id, None)
        if rel is None:
            return False

        try:
            graph.remove_edge(rel.from_entity, rel.to_entity, key=relation_id)
        except nx.NetworkXError:
            pass

        # Clean up isolated nodes (nodes with no edges)
        for node_id in (rel.from_entity, rel.to_entity):
            if graph.has_node(node_id) and graph.degree(node_id) == 0:
                graph.remove_node(node_id)

        db.delete_relation_row(relation_id)
    logger.info("Removed relation: %s", relation_id)
    return True


def get_relation(relation_id: str) -> Relation | None:
    """Get a relation by ID."""
    with GRAPH_LOCK:
        _get_graph()  # ensure loaded
        return _relations.get(relation_id)


def get_relations_for_entity(entity_id: str) -> list[Relation]:
    """Get all relations involving an entity (incoming and outgoing)."""
    with GRAPH_LOCK:
        _get_graph()  # ensure loaded
        return [
            r for r in _relations.values()
            if r.from_entity == entity_id or r.to_entity == entity_id
        ]


def remove_entity_relations(entity_id: str) -> int:
    """Remove all relations for an entity. Returns count removed."""
    with GRAPH_LOCK:
        relations = get_relations_for_entity(entity_id)
        for rel in relations:
            remove_relation(rel.id)
        return len(relations)


def get_graph() -> nx.MultiDiGraph:
    """Get the graph instance (read-only access).

    Callers that iterate the returned graph while writes may be in flight must
    hold GRAPH_LOCK for the duration of the traversal.
    """
    return _get_graph()


def get_relation_count() -> int:
    """Count total relations."""
    with GRAPH_LOCK:
        _get_graph()  # ensure loaded
        return len(_relations)


def get_all_relations() -> list[Relation]:
    """Get all relations (a snapshot list, safe to iterate)."""
    with GRAPH_LOCK:
        _get_graph()  # ensure loaded
        return list(_relations.values())
