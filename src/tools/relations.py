"""Relation CRUD tool implementations."""

import logging

from src.indexer.store import resolve_entity
from src.graph.manager import add_relation, remove_relation, get_relation
from src.models.relation import Relation, RELATION_TYPES

logger = logging.getLogger(__name__)


def _generate_id() -> str:
    import uuid
    return uuid.uuid4().hex[:12]


def tool_create_relation(from_entity: str, to_entity: str,
                         relation_type: str, vault: str = "",
                         weight: float = 1.0, context: str = "") -> str:
    """Create a directed relation between two entities.

    Args:
        from_entity: Source entity name or ID.
        to_entity: Target entity name or ID.
        relation_type: Relation type (depends_on, solves, related_to,
                       contradicts, builds_on, applies_to, caused_by,
                       part_of, uses, created_by, maintained_by, etc.).
        vault: Vault name (helps disambiguate entity names).
        weight: Relation strength (0.0 to 1.0, default 1.0).
        context: Optional context/description for the relation.

    Returns:
        Confirmation or error.
    """
    from_ent = resolve_entity(from_entity, vault or None)
    if from_ent is None:
        return f"Source entity not found: '{from_entity}'"

    to_ent = resolve_entity(to_entity, vault or None)
    if to_ent is None:
        return f"Target entity not found: '{to_entity}'"

    rel = Relation(
        id=_generate_id(),
        from_entity=from_ent.id,
        to_entity=to_ent.id,
        relation_type=relation_type.strip().lower(),
        weight=max(0.0, min(1.0, weight)),
        context=context,
    )
    add_relation(rel)

    return (f"Relation created: {from_ent.name} -[{rel.relation_type}]-> {to_ent.name}\n"
            f"  ID: {rel.id}")


def tool_delete_relation(relation_id: str) -> str:
    """Remove a relation by ID.

    Args:
        relation_id: The relation ID to delete.

    Returns:
        Confirmation or error.
    """
    if remove_relation(relation_id):
        return f"Relation deleted: {relation_id}"
    return f"Relation not found: '{relation_id}'"
