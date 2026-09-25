"""Relation CRUD tool implementations."""

import logging
import re

from src.indexer.store import get_entity, resolve_entity
from src.graph.manager import (add_relation, remove_relation, get_relation,
                               update_relation)
from src.models.relation import Relation, RELATION_TYPES, canonicalize_relation_type

logger = logging.getLogger(__name__)

# A context carrying a year is almost always a fact rather than a
# disambiguator. Relation context is invisible to search and has no
# supersession path, so such claims rot silently — warn, don't refuse.
_DATED_CONTEXT_RE = re.compile(r"\b(?:19|20)\d{2}\b")


def _generate_id() -> str:
    import uuid
    return uuid.uuid4().hex[:12]


def _context_warning(context: str) -> str:
    """Flag a context that looks like a stored fact instead of a label."""
    if not _DATED_CONTEXT_RE.search(context or ""):
        return ""
    return (
        "\n  Note: this context contains a date, so it reads as a fact rather "
        "than a disambiguator. Relation context never appears in search "
        "results and cannot be superseded, so a dated claim here goes stale "
        "invisibly. Record the fact as an observation on the entity it is "
        "about, and keep the context to what distinguishes this edge."
    )


def tool_create_relation(from_entity: str, to_entity: str,
                         relation_type: str, vault: str = "",
                         weight: float = 1.0, context: str = "") -> str:
    """Create a directed relation between two entities.

    relation_type must be one of the canonical types; common synonyms are
    auto-canonicalized (direction-flipped forms like created_by swap the
    endpoints automatically). If nothing fits, use related_to and put the
    nuance in context.

    context is a disambiguator ("the 2024 contract", "which Alex"), not a
    fact store. It never appears in search results and cannot be superseded,
    so a fact written here rots where nothing can see or correct it. Facts
    belong on the entity they are about, as observations.

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
            f"  ID: {rel.id}{note}{_context_warning(context)}")


def tool_update_relation(relation_id: str, context: str | None = None,
                         weight: float | None = None) -> str:
    """Edit an existing relation's context or weight in place.

    The relation keeps its ID, endpoints, type and creation date — use this
    to correct a stale context rather than deleting and recreating the edge,
    which loses both. Pass context="" to clear it.

    Args:
        relation_id: The relation ID to update.
        context: New context, or None to leave unchanged. "" clears it.
        weight: New weight (0.0 to 1.0), or None to leave unchanged.

    Returns:
        Confirmation or error.
    """
    if context is None and weight is None:
        return "Nothing to update: pass context and/or weight."

    before = get_relation(relation_id)
    if before is None:
        return f"Relation not found: '{relation_id}'"
    old_context = before.context

    rel = update_relation(relation_id, context=context, weight=weight)
    if rel is None:
        return f"Relation not found: '{relation_id}'"

    from_ent = get_entity(rel.from_entity)
    to_ent = get_entity(rel.to_entity)
    from_name = from_ent.name if from_ent else rel.from_entity
    to_name = to_ent.name if to_ent else rel.to_entity

    lines = [f"Relation updated: {from_name} -[{rel.relation_type}]-> {to_name}",
             f"  ID: {rel.id}"]
    if context is not None:
        lines.append(f"  context: {old_context or '(empty)'} -> {rel.context or '(empty)'}")
    if weight is not None:
        lines.append(f"  weight: {rel.weight}")
    return "\n".join(lines) + _context_warning(rel.context)


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
