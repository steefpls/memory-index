"""Entity and observation CRUD tool implementations."""

import json
import logging

from src.config import VAULTS, get_vault, create_vault as config_create_vault
from src.indexer.store import (
    create_entity, get_entity, get_entity_by_name, update_entity,
    delete_entity, list_entities, resolve_entity,
    add_observation, get_observations, delete_observation,
)
from src.graph.manager import get_relations_for_entity, remove_entity_relations
from src.models.entity import ENTITY_TYPES

logger = logging.getLogger(__name__)


def tool_create_entity(name: str, entity_type: str, vault: str,
                       observations: str = "", source: str = "") -> str:
    """Create a named entity with optional initial observations.

    Args:
        name: Entity name (e.g., "Python", "memory-index", "Alice").
        entity_type: Type (person, project, concept, decision, error,
                     solution, technology, event, organization, etc.).
        vault: Vault to store in. Created automatically if it doesn't exist.
        observations: Pipe-separated observations (e.g., "Fact 1|Fact 2|Fact 3").
        source: Optional source attribution for observations.

    Returns:
        Confirmation with entity details.
    """
    if not name or not name.strip():
        return "Error: name is required."
    if not entity_type or not entity_type.strip():
        return "Error: entity_type is required."
    if not vault or not vault.strip():
        return "Error: vault is required."

    # Auto-create vault if needed
    if vault not in VAULTS:
        config_create_vault(vault)

    obs_list = [o.strip() for o in observations.split("|") if o.strip()] if observations else None

    entity = create_entity(name.strip(), entity_type.strip().lower(), vault.strip(),
                          observations=obs_list, source=source)

    obs_count = len(get_observations(entity.id))
    return (f"Entity created: {entity.name} ({entity.entity_type})\n"
            f"  ID: {entity.id}\n"
            f"  Vault: {entity.vault}\n"
            f"  Observations: {obs_count}")


def tool_get_entity(name_or_id: str, vault: str = "",
                    offset: int = 0, limit: int = 30,
                    full: bool = False,
                    include_superseded: bool = False) -> str:
    """Get entity details with observations and relations.

    By default returns header + counts + all relations + the `limit` most
    recent active observations. Set full=True to dump everything in one call.
    Superseded observations are hidden by default; pass include_superseded=True
    to also list them (for full history use the temporal tools).

    Args:
        name_or_id: Entity name or ID.
        vault: Vault name (helps disambiguate names across vaults).
        offset: Skip this many active observations (newest-first ordering).
        limit: Max active observations to show (default 30, ignored if full=True).
        full: If True, return every active observation, ignoring offset/limit.
        include_superseded: If True, also list superseded observations.

    Returns:
        Entity details with observations and relations.
    """
    entity = resolve_entity(name_or_id, vault or None)
    if entity is None:
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

    lines = [
        f"Entity: {entity.name} ({entity.entity_type})",
        f"  ID: {entity.id}",
        f"  Vault: {entity.vault}",
        f"  Created: {entity.created_at}",
        f"  Updated: {entity.updated_at}",
        f"  Observations: {obs_total} active"
        + (f" + {len(superseded_only)} superseded" if superseded_only else ""),
        f"  Relations: {len(relations)}",
        "",
    ]

    if shown_obs:
        if full or obs_total <= len(shown_obs):
            lines.append(f"Observations ({obs_total}, newest first):")
        else:
            end = offset_used + len(shown_obs)
            lines.append(
                f"Observations (showing {offset_used + 1}-{end} of {obs_total}, "
                f"newest first; use offset={end} for next page or full=True to "
                f"dump all):"
            )
        for obs in shown_obs:
            src = f" [source: {obs.source}]" if obs.source else ""
            lines.append(f"  - {obs.content}{src} (id: {obs.id})")
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
            lines.append(
                f"  - [old] {obs.content} (id: {obs.id} -> {obs.superseded_by})"
            )
        lines.append("")

    if relations:
        lines.append(f"Relations ({len(relations)}):")
        for rel in relations:
            arrow = "->" if rel.from_entity == entity.id else "<-"
            other = rel.to_entity if rel.from_entity == entity.id else rel.from_entity
            ctx = f" - {rel.context}" if rel.context else ""
            lines.append(f"  {arrow} {other} [{rel.relation_type}]{ctx}")

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
                         supersedes: str = "") -> str:
    """Add an observation (fact) to an existing entity.

    Args:
        name_or_id: Entity name or ID.
        content: The observation text.
        vault: Vault name (helps disambiguate names).
        source: Optional source attribution.
        supersedes: Optional observation ID that this replaces. The old
                    observation is kept for history but excluded from search.

    Returns:
        Confirmation or error.
    """
    entity = resolve_entity(name_or_id, vault or None)
    if entity is None:
        return f"Entity not found: '{name_or_id}'"

    obs = add_observation(entity.id, content, source=source,
                          supersedes=supersedes)
    if obs is None:
        return "Error: failed to add observation."

    msg = (f"Observation added to '{entity.name}':\n"
           f"  {content}\n"
           f"  ID: {obs.id}")
    if supersedes:
        msg += f"\n  Supersedes: {supersedes}"
    return msg


def tool_delete_observation(observation_id: str) -> str:
    """Remove an observation by ID.

    Args:
        observation_id: The observation ID to delete.

    Returns:
        Confirmation or error.
    """
    if delete_observation(observation_id):
        return f"Observation deleted: {observation_id}"
    return f"Observation not found: '{observation_id}'"
