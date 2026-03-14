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


def tool_get_entity(name_or_id: str, vault: str = "") -> str:
    """Get entity details including observations and relations.

    Args:
        name_or_id: Entity name or ID.
        vault: Vault name (helps disambiguate names across vaults).

    Returns:
        Entity details with observations and relations.
    """
    entity = resolve_entity(name_or_id, vault or None)
    if entity is None:
        return f"Entity not found: '{name_or_id}'"

    obs_list = get_observations(entity.id)
    superseded_list = get_observations(entity.id, include_superseded=True)
    superseded_only = [o for o in superseded_list if o.is_superseded]
    relations = get_relations_for_entity(entity.id)

    lines = [
        f"Entity: {entity.name} ({entity.entity_type})",
        f"  ID: {entity.id}",
        f"  Vault: {entity.vault}",
        f"  Created: {entity.created_at}",
        f"  Updated: {entity.updated_at}",
        "",
    ]

    if obs_list:
        lines.append(f"Observations ({len(obs_list)}):")
        for obs in obs_list:
            src = f" [source: {obs.source}]" if obs.source else ""
            lines.append(f"  - {obs.content}{src}")
            lines.append(f"    ID: {obs.id}")
        lines.append("")

    if superseded_only:
        lines.append(f"Superseded observations ({len(superseded_only)}):")
        for obs in superseded_only:
            lines.append(f"  - [old] {obs.content}")
            lines.append(f"    ID: {obs.id} -> replaced by {obs.superseded_by}")
        lines.append("")

    if relations:
        lines.append(f"Relations ({len(relations)}):")
        for rel in relations:
            if rel.from_entity == entity.id:
                lines.append(f"  -> {rel.to_entity} [{rel.relation_type}]")
            else:
                lines.append(f"  <- {rel.from_entity} [{rel.relation_type}]")
            if rel.context:
                lines.append(f"     Context: {rel.context}")

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

    # Remove relations too
    rel_count = remove_entity_relations(entity.id)
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
