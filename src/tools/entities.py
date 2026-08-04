"""Entity and observation CRUD tool implementations."""

import json
import logging

from src.config import VAULTS, get_vault, create_vault as config_create_vault
from src.indexer.store import (
    create_entity, get_entity, get_entity_by_name, update_entity,
    delete_entity, list_entities, resolve_entity,
    add_observation, add_observations, get_observations, delete_observation,
    delete_observation_detailed, undelete_observation,
)
from src.graph.manager import get_relations_for_entity, remove_entity_relations
from src.models.entity import ENTITY_TYPES

logger = logging.getLogger(__name__)


def _coerce_str_list(value) -> list[str]:
    """Normalize a list-of-strings param into a clean list.

    Accepts a real list (the supported form) or a bare string, which is
    treated as ONE item. Deliberately does NOT split on any delimiter — a
    fact containing '|' must survive intact.
    """
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    return [str(v).strip() for v in value if str(v).strip()]


def _validate_entity_type(entity_type: str) -> str | None:
    """Return an error string if the type is not canonical, else None."""
    if entity_type.strip().lower() not in ENTITY_TYPES:
        valid = ", ".join(sorted(ENTITY_TYPES))
        return (f"Error: unknown entity_type '{entity_type}'. "
                f"Use one of: {valid}.")
    return None


def tool_create_entity(name: str, entity_type: str, vault: str,
                       observations: list[str] | None = None,
                       source: str = "") -> str:
    """Create a named entity with optional initial observations.

    Args:
        name: Entity name (e.g., "Python", "memory-index", "Alice").
        entity_type: Canonical type — one of: person, project, organization,
                     technology, concept, decision, event, preference,
                     pattern, error, solution, reference, location, process,
                     artifact.
        vault: Vault to store in. Created automatically if it doesn't exist.
        observations: List of observation strings, one atomic fact per item
                      (e.g., ["Fact 1", "Fact 2"]). Never split — a fact may
                      contain any character, including '|'.
        source: Optional source attribution for observations.

    Returns:
        Confirmation with entity details.
    """
    if not name or not name.strip():
        return "Error: name is required."
    if not entity_type or not entity_type.strip():
        return "Error: entity_type is required."
    type_error = _validate_entity_type(entity_type)
    if type_error:
        return type_error
    if not vault or not vault.strip():
        return "Error: vault is required."

    # Auto-create vault if needed
    if vault not in VAULTS:
        config_create_vault(vault)

    obs_list = _coerce_str_list(observations) or None

    entity = create_entity(name.strip(), entity_type.strip().lower(), vault.strip(),
                          observations=obs_list, source=source)

    obs_count = len(get_observations(entity.id))
    return (f"Entity created: {entity.name} ({entity.entity_type})\n"
            f"  ID: {entity.id}\n"
            f"  Vault: {entity.vault}\n"
            f"  Observations: {obs_count}")


def _obs_json(obs) -> dict:
    """Serialize an observation for output_format='json'."""
    return {
        "id": obs.id,
        "content": obs.content,
        "source": obs.source or None,
        "created_at": obs.created_at,
        "occurred_at": obs.occurred_at,
        "superseded": obs.is_superseded,
        "superseded_by": obs.superseded_by or None,
        "superseded_at": obs.superseded_at,
    }


def tool_get_entity(name_or_id: str, vault: str = "",
                    offset: int = 0, limit: int = 10,
                    full: bool = False,
                    include_superseded: bool = False,
                    show_ids: bool = False,
                    output_format: str = "text") -> str:
    """Get entity details with observations and relations.

    By default returns header + counts + all relations + the `limit` most
    recent active observations. Set full=True to dump everything in one call.
    Observation IDs are hidden by default; pass show_ids=True when you need
    them (e.g. to supersede or delete). Superseded observations are hidden by
    default; pass include_superseded=True to also list them (for full history
    use the temporal tools).

    output_format='json' returns the same data as a machine-readable object —
    every observation carries its ID (regardless of show_ids) and every
    relation carries its relation ID, which the text form never prints. Use it
    when the caller intends to act on individual rows.

    Args:
        name_or_id: Entity name or ID.
        vault: Vault name (helps disambiguate names across vaults).
        offset: Skip this many active observations (newest-first ordering).
        limit: Max active observations to show (default 10, ignored if full=True).
        full: If True, return every active observation, ignoring offset/limit.
        include_superseded: If True, also list superseded observations.
        show_ids: If True, append observation IDs inline (for supersede/delete).
        output_format: "text" (default) or "json".

    Returns:
        Entity details with observations and relations.
    """
    as_json = output_format.strip().lower() == "json"
    entity = resolve_entity(name_or_id, vault or None)
    if entity is None:
        if as_json:
            return json.dumps({
                "error": "not_found",
                "message": f"Entity not found: '{name_or_id}'",
            })
        return f"Entity not found: '{name_or_id}'"

    obs_active = get_observations(entity.id)
    obs_active.sort(key=lambda o: o.created_at, reverse=True)
    obs_total = len(obs_active)

    superseded_only: list = []
    if include_superseded:
        all_obs = get_observations(entity.id, include_superseded=True)
        superseded_only = [o for o in all_obs if o.is_superseded]
        superseded_only.sort(key=lambda o: o.created_at, reverse=True)

    relations = get_relations_for_entity(entity.id)

    if full:
        offset_used = 0
        shown_obs = obs_active
    else:
        offset_used = max(0, offset)
        limit_used = max(1, limit)
        shown_obs = obs_active[offset_used:offset_used + limit_used]

    if as_json:
        from src.indexer.store import get_entity as _store_get_entity
        rel_json = []
        for rel in relations:
            outgoing = rel.from_entity == entity.id
            other_id = rel.to_entity if outgoing else rel.from_entity
            other_ent = _store_get_entity(other_id)
            rel_json.append({
                "id": rel.id,
                "direction": "out" if outgoing else "in",
                "other_id": other_id,
                "other_name": other_ent.name if other_ent else other_id,
                "type": rel.relation_type,
                "context": rel.context or None,
                "weight": rel.weight,
            })
        return json.dumps({
            "entity": {
                "id": entity.id,
                "name": entity.name,
                "type": entity.entity_type,
                "vault": entity.vault,
                "created_at": entity.created_at,
                "updated_at": entity.updated_at,
            },
            "observations": [_obs_json(o) for o in shown_obs],
            "observations_total": obs_total,
            "observations_offset": offset_used,
            "superseded": [_obs_json(o) for o in superseded_only],
            "relations": rel_json,
        })

    obs_summary = f"{obs_total} active obs"
    if superseded_only:
        obs_summary += f" (+{len(superseded_only)} superseded)"
    lines = [
        f"Entity: {entity.name} ({entity.entity_type}) · "
        f"id={entity.id} · vault={entity.vault}",
        f"  {obs_summary} · {len(relations)} relations · "
        f"updated {entity.updated_at}",
        "",
    ]

    if shown_obs:
        if full or obs_total <= len(shown_obs):
            lines.append(f"Observations ({obs_total}, newest first):")
        else:
            end = offset_used + len(shown_obs)
            remaining = obs_total - end
            lines.append(
                f"Observations ({offset_used + 1}-{end}/{obs_total}, "
                f"+{remaining} more, offset={end}):"
            )
        for obs in shown_obs:
            src = f" [src: {obs.source}]" if obs.source else ""
            tail = f" (id: {obs.id})" if show_ids else ""
            lines.append(f"  - {obs.content}{src}{tail}")
        lines.append("")
    elif obs_total > 0 and offset_used >= obs_total:
        lines.append(
            f"Observations: offset={offset_used} is past the end "
            f"({obs_total} total)."
        )
        lines.append("")

    if superseded_only:
        lines.append(f"Superseded observations ({len(superseded_only)}):")
        for obs in superseded_only:
            tail = f" (id: {obs.id} -> {obs.superseded_by})" if show_ids else ""
            lines.append(f"  - [old] {obs.content}{tail}")
        lines.append("")

    if relations:
        # Resolve names once so the model can act on them without a second lookup.
        from src.indexer.store import get_entity as _store_get_entity
        lines.append(f"Relations ({len(relations)}):")
        for rel in relations:
            arrow = "->" if rel.from_entity == entity.id else "<-"
            other_id = rel.to_entity if rel.from_entity == entity.id else rel.from_entity
            other_ent = _store_get_entity(other_id)
            other_name = other_ent.name if other_ent else other_id
            ctx = f" — {rel.context}" if rel.context else ""
            lines.append(f"  {arrow} {other_name} [{rel.relation_type}]{ctx}")

    return "\n".join(lines)


def tool_update_entity(name_or_id: str, new_name: str = "",
                       new_type: str = "", vault: str = "") -> str:
    """Update an entity's name or type.

    Args:
        name_or_id: Current entity name or ID.
        new_name: New name (leave empty to keep current).
        new_type: New type (leave empty to keep current).
        vault: Vault name (helps disambiguate names).

    Returns:
        Confirmation or error.
    """
    if new_type:
        type_error = _validate_entity_type(new_type)
        if type_error:
            return type_error

    entity = resolve_entity(name_or_id, vault or None)
    if entity is None:
        return f"Entity not found: '{name_or_id}'"

    updated = update_entity(
        entity.id,
        name=new_name.strip() if new_name else None,
        entity_type=new_type.strip().lower() if new_type else None,
    )
    if updated is None:
        return "Error: update failed."

    return f"Entity updated: {updated.name} ({updated.entity_type}), ID: {updated.id}"


def tool_delete_entity(name_or_id: str, vault: str = "") -> str:
    """Soft delete an entity and its observations.

    Args:
        name_or_id: Entity name or ID.
        vault: Vault name (helps disambiguate names).

    Returns:
        Confirmation or error.
    """
    entity = resolve_entity(name_or_id, vault or None)
    if entity is None:
        return f"Entity not found: '{name_or_id}'"

    # Count relations before delete (delete_entity now cleans them up itself)
    rel_count = len(get_relations_for_entity(entity.id))
    success = delete_entity(entity.id)
    if not success:
        return "Error: delete failed."

    return f"Deleted entity '{entity.name}' (ID: {entity.id}), removed {rel_count} relations."


def tool_list_entities(vault: str = "", entity_type: str = "",
                       offset: int = 0, limit: int = 20) -> str:
    """List entities with optional filters.

    Args:
        vault: Filter by vault (empty = all vaults).
        entity_type: Filter by type (empty = all types).
        offset: Pagination offset.
        limit: Max results (1-50).

    Returns:
        Paginated entity list.
    """
    limit = min(max(limit, 1), 50)
    entities, total = list_entities(
        vault=vault or None,
        entity_type=entity_type or None,
        offset=offset,
        limit=limit,
    )

    if not entities:
        return "No entities found."

    lines = [f"Entities ({total} total, showing {offset + 1}-{offset + len(entities)}):"]
    for ent in entities:
        obs_count = len(get_observations(ent.id))
        lines.append(f"  {ent.name} ({ent.entity_type}) [{ent.vault}] - {obs_count} observations")
        lines.append(f"    ID: {ent.id}")

    if offset + len(entities) < total:
        lines.append(f"\nUse offset={offset + len(entities)} for next page.")

    return "\n".join(lines)


def tool_add_observation(name_or_id: str, content: str,
                         vault: str = "", source: str = "",
                         supersedes: str = "",
                         occurred_at: str = "") -> str:
    """Add an observation (fact) to an existing entity.

    Args:
        name_or_id: Entity name or ID.
        content: The observation text.
        vault: Vault name (helps disambiguate names).
        source: Optional source attribution.
        supersedes: Optional observation ID that this replaces. The old
                    observation is kept for history but excluded from search.
        occurred_at: Optional ISO date/datetime for when the fact actually
                     happened. Defaults to empty, meaning "use ingestion time".

    Returns:
        Confirmation or error.
    """
    entity = resolve_entity(name_or_id, vault or None)
    if entity is None:
        return f"Entity not found: '{name_or_id}'"

    obs = add_observation(entity.id, content, source=source,
                          supersedes=supersedes, occurred_at=occurred_at)
    if obs is None:
        return "Error: failed to add observation."

    msg = f"Observation added to '{entity.name}': id={obs.id}"
    if supersedes:
        msg += f", supersedes={supersedes}"
    if obs.occurred_at:
        msg += f", occurred_at={obs.occurred_at}"
    return msg


def tool_add_observations(name_or_id: str, contents: list[str],
                          vault: str = "", source: str = "",
                          occurred_at: list[str] | None = None) -> str:
    """Add multiple observations to a single entity in one call.

    Each list item becomes its own observation (one embedding per fact —
    atomicity preserved per CLAUDE.md), but the MCP round-trip and the
    embed/save work are collapsed to one batch.

    Args:
        name_or_id: Entity name or ID.
        contents: List of observation strings, one atomic fact per item
                  (e.g., ["Fact 1", "Fact 2"]). Contents are never split, so a
                  fact containing '|' is stored verbatim.
        vault: Vault name (helps disambiguate names).
        source: Optional source attribution applied to all added observations.
        occurred_at: Optional parallel list of ISO date/datetime strings (event
                     time), same length as contents. Use "" for items with no
                     known event time. Omit entirely to use ingestion time.

    Returns:
        Confirmation listing each added observation ID, or error.
    """
    items = _coerce_str_list(contents)
    if not items:
        return "Error: contents is required (a non-empty list of strings)."

    # Validate occurred_at against the RAW contents, before blanks are dropped,
    # so the caller's parallel-list indices are the ones being checked.
    when_list: list[str] | None = None
    if occurred_at is not None:
        raw = contents if isinstance(contents, list) else [contents]
        if isinstance(occurred_at, str):
            occurred_at = [occurred_at]
        if len(occurred_at) != len(raw):
            return (
                f"Error: occurred_at has {len(occurred_at)} entries but contents "
                f"has {len(raw)} — they must match (or omit occurred_at)."
            )
        # Re-pair after dropping blank contents.
        when_list = [
            (occurred_at[i] or "")
            for i, c in enumerate(raw)
            if str(c or "").strip()
        ]

    entity = resolve_entity(name_or_id, vault or None)
    if entity is None:
        return f"Entity not found: '{name_or_id}'"

    created = add_observations(entity.id, items, source=source,
                               occurred_at=when_list)
    if not created:
        return "Error: failed to add observations."

    lines = [f"Added {len(created)} observations to '{entity.name}':"]
    for obs in created:
        when = f"  occurred_at={obs.occurred_at}" if obs.occurred_at else ""
        lines.append(f"  id={obs.id}{when}")
    return "\n".join(lines)


def tool_delete_observation(observation_id: str) -> str:
    """Remove an observation by ID.

    The delete is soft — undelete_observation restores it. Anything this
    observation superseded is revived, so deleting a correction puts the fact
    it corrected back in play.

    Args:
        observation_id: The observation ID to delete.

    Returns:
        Confirmation or error.
    """
    ok, revived = delete_observation_detailed(observation_id)
    if not ok:
        return f"Observation not found: '{observation_id}'"
    msg = f"Observation deleted: {observation_id}"
    if revived:
        msg += f", revived {len(revived)} superseded: {', '.join(revived)}"
    return msg


def tool_undelete_observation(observation_id: str) -> str:
    """Restore a deleted observation by ID.

    Args:
        observation_id: The observation ID to restore.

    Returns:
        Confirmation or error.
    """
    obs = undelete_observation(observation_id)
    if obs is None:
        return (f"Cannot undelete '{observation_id}': unknown ID, not "
                f"deleted, or its entity is gone.")
    state = " (still superseded)" if obs.is_superseded else ""
    return f"Observation restored: {observation_id}{state}"
