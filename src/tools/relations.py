"""Relation CRUD tool implementations."""

import logging

from src.indexer.store import resolve_entity
from src.graph.manager import add_relation, remove_relation, get_relation
from src.models.relation import Relation, RELATION_TYPES, canonicalize_relation_type

logger = logging.getLogger(__name__)


def _generate_id() -> str:
    import uuid
    return uuid.uuid4().hex[:12]


def tool_create_relation(from_entity: str, to_entity: str,
                         relation_type: str, vault: str = "",
                         weight: float = 1.0, context: str = "") -> str:
    """Create a directed relation between two entities.

    relation_type must be one of the canonical types; common synonyms are
    auto-canonicalized (direction-flipped forms like created_by swap the
    endpoints automatically). If nothing fits, use related_to and put the
    nuance in context.

    Canonical types: related_to; part_of, uses, depends_on, involves,
    applies_to, builds_on, replaces, created, maintains; works_at, worked_at,
    works_on, reports_to, leads, founded, funds, friend_of,
    collaborates_with, learned_from, participated_in; solves, caused_by,
    contradicts, blocks.

    Args:
        from_entity: Source entity name or ID.
        to_entity: Target entity name or ID.
        relation_type: Canonical relation type (or a known synonym).
        vault: Vault name (helps disambiguate entity names).
        weight: Relation strength (0.0 to 1.0, default 1.0).
        context: Optional context/description for the relation.

    Returns:
        Confirmation or error.
    """
    resolved = canonicalize_relation_type(relation_type)
    if resolved is None:
        valid = ", ".join(sorted(RELATION_TYPES))
        return (
            f"Error: unknown relation_type '{relation_type}'. "
            f"Use one of: {valid}. "
            f"If nothing fits, use 'related_to' and describe the nuance in "
            f"the context parameter."
        )
    canonical_type, flip = resolved

    from_ent = resolve_entity(from_entity, vault or None)
    if from_ent is None:
        return f"Source entity not found: '{from_entity}'"

    to_ent = resolve_entity(to_entity, vault or None)
    if to_ent is None:
        return f"Target entity not found: '{to_entity}'"

    if flip:
        from_ent, to_ent = to_ent, from_ent

    rel = Relation(
        id=_generate_id(),
        from_entity=from_ent.id,
        to_entity=to_ent.id,
        relation_type=canonical_type,
        weight=max(0.0, min(1.0, weight)),
        context=context,
    )
    add_relation(rel)

    note = ""
    normalized = (relation_type or "").strip().lower()
    if normalized != canonical_type:
        note = (f"\n  (canonicalized from '{normalized}'"
                + (", direction flipped" if flip else "") + ")")
    return (f"Relation created: {from_ent.name} -[{rel.relation_type}]-> {to_ent.name}\n"
            f"  ID: {rel.id}{note}")


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
