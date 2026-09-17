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


def update_relation(relation_id: str, context: str | None = None,
                    weight: float | None = None) -> Relation | None:
    """Edit a relation's context or weight in place.

    The id, endpoints, type and created_at are preserved — this exists so a
    context can be corrected without the delete-and-recreate that throws the
    edge's identity and creation date away. Returns the updated relation, or
    None if no such relation.
    """
    with GRAPH_LOCK:
        graph = _get_graph()
        rel = _relations.get(relation_id)
        if rel is None:
            return None

        if context is not None:
            rel.context = context
        if weight is not None:
            rel.weight = max(0.0, min(1.0, weight))

        edge = graph.get_edge_data(rel.from_entity, rel.to_entity, key=relation_id)
        if edge is not None:
            edge["context"] = rel.context
            edge["weight"] = rel.weight

        db.upsert_relations([rel])
    logger.info("Updated relation: %s", relation_id)
    return rel


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


def repoint_entity_relations(source_id: str, target_id: str) -> tuple[int, int]:
    """Re-point every relation touching `source_id` onto `target_id`.

    Used by entity merge: outgoing `source -> X` becomes `target -> X`,
    incoming `X -> source` becomes `X -> target`, and a `source -> source`
    self-loop becomes `target -> target`. Relation IDs, types, weights,
    contexts and creation dates are preserved — only the endpoints move.

    Dedupe is by (from, to, type): when the re-pointed signature already
    exists (on the target or among already re-pointed edges), the duplicate
    is hard-removed instead of creating a parallel edge.

    Returns (moved, deduped).
    """
    with GRAPH_LOCK:
        graph = _get_graph()  # ensure loaded
        source_rels = [
            r for r in _relations.values()
            if r.from_entity == source_id or r.to_entity == source_id
        ]
        if not source_rels:
            return (0, 0)

        existing = {
            (r.from_entity, r.to_entity, r.relation_type)
            for r in _relations.values()
            if r.from_entity != source_id and r.to_entity != source_id
        }

        moved = 0
        deduped = 0
        to_update: list = []
        to_remove_ids: list[str] = []

        for rel in source_rels:
            new_from = target_id if rel.from_entity == source_id else rel.from_entity
            new_to = target_id if rel.to_entity == source_id else rel.to_entity
            sig = (new_from, new_to, rel.relation_type)
            if sig in existing:
                to_remove_ids.append(rel.id)
                deduped += 1
                continue

            old_from, old_to = rel.from_entity, rel.to_entity
            try:
                graph.remove_edge(old_from, old_to, key=rel.id)
            except nx.NetworkXError:
                pass
            rel.from_entity = new_from
            rel.to_entity = new_to
            graph.add_edge(
                new_from, new_to,
                key=rel.id,
                relation_type=rel.relation_type,
                weight=rel.weight,
                context=rel.context,
                created_at=rel.created_at,
            )
            to_update.append(rel)
            existing.add(sig)
            moved += 1

        for rid in to_remove_ids:
            rel = _relations.pop(rid, None)
            if rel is None:
                continue
            try:
                graph.remove_edge(rel.from_entity, rel.to_entity, key=rid)
            except nx.NetworkXError:
                pass
            for node_id in (rel.from_entity, rel.to_entity):
                if graph.has_node(node_id) and graph.degree(node_id) == 0:
                    graph.remove_node(node_id)

        if to_update:
            db.upsert_relations(to_update)
        for rid in to_remove_ids:
            db.delete_relation_row(rid)

        if graph.has_node(source_id) and graph.degree(source_id) == 0:
            graph.remove_node(source_id)

        return (moved, deduped)


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
