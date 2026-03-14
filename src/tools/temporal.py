"""Temporal query tools — timeline, point-in-time, and temporal neighbor queries."""

import json
import logging
from datetime import datetime, timezone

from src.config import VAULTS
from src.indexer.store import (
    resolve_entity, get_entity, get_observations,
    _load_store, _entities, _observations,
)
from src.graph.traversal import get_neighbors

logger = logging.getLogger(__name__)


def _parse_iso(dt_str: str) -> datetime | None:
    """Parse an ISO date or datetime string. Always returns UTC-aware. None on failure."""
    if not dt_str:
        return None
    try:
        dt = datetime.fromisoformat(dt_str)
        # Ensure timezone-aware (default to UTC if naive)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        pass
    try:
        return datetime.strptime(dt_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _obs_created_dt(obs) -> datetime | None:
    """Get observation created_at as datetime."""
    return _parse_iso(obs.created_at)


def tool_query_timeline(vault: str = "", start: str = "", end: str = "",
                        entity_type: str = "", limit: int = 50,
                        output_format: str = "text") -> str:
    """Query observations across a time range, ordered chronologically.

    Returns a timeline of facts/events within the specified window.

    Args:
        vault: Vault to query (empty = all vaults).
        start: Start date/datetime (ISO format, inclusive). Empty = no lower bound.
        end: End date/datetime (ISO format, exclusive). Empty = no upper bound.
        entity_type: Optional filter by entity type.
        limit: Max results (default 50, max 200).
        output_format: "text" (default) or "json".
    """
    output_format = (output_format or "text").lower()
    if output_format not in {"text", "json"}:
        return "Error: output_format must be 'text' or 'json'."

    limit = min(max(limit, 1), 200)

    start_dt = _parse_iso(start)
    end_dt = _parse_iso(end)

    if start and start_dt is None:
        return f"Error: invalid start date '{start}'. Use ISO format (YYYY-MM-DD or full ISO datetime)."
    if end and end_dt is None:
        return f"Error: invalid end date '{end}'. Use ISO format (YYYY-MM-DD or full ISO datetime)."

    _load_store()

    # Determine which entity IDs are in scope
    vault_filter = vault or None
    type_filter = entity_type.strip().lower() if entity_type else None

    entity_ids_in_scope = set()
    for ent in _entities.values():
        if ent.deleted:
            continue
        if vault_filter and ent.vault != vault_filter:
            continue
        if type_filter and ent.entity_type != type_filter:
            continue
        entity_ids_in_scope.add(ent.id)

    if not entity_ids_in_scope:
        return "No entities match the specified filters."

    # Collect observations in the time window
    timeline_items = []
    for obs in _observations.values():
        if obs.deleted or obs.is_superseded:
            continue
        if obs.entity_id not in entity_ids_in_scope:
            continue

        obs_dt = _obs_created_dt(obs)
        if obs_dt is None:
            continue
        if start_dt and obs_dt < start_dt:
            continue
        if end_dt and obs_dt >= end_dt:
            continue

        ent = _entities.get(obs.entity_id)
        if ent is None:
            continue

        timeline_items.append({
            "observation_id": obs.id,
            "entity_id": ent.id,
            "entity_name": ent.name,
            "entity_type": ent.entity_type,
            "vault": ent.vault,
            "content": obs.content,
            "source": obs.source,
            "created_at": obs.created_at,
        })

    # Sort chronologically
    timeline_items.sort(key=lambda x: x["created_at"])
    timeline_items = timeline_items[:limit]

    if not timeline_items:
        date_range = ""
        if start:
            date_range += f" after {start}"
        if end:
            date_range += f" before {end}"
        return f"No observations found{date_range}."

    if output_format == "json":
        return json.dumps({
            "timeline": timeline_items,
            "count": len(timeline_items),
            "start": start or None,
            "end": end or None,
        }, indent=2)

    # Text format
    lines = [f"Timeline ({len(timeline_items)} observations):"]
    current_date = ""
    for item in timeline_items:
        # Group by date
        item_date = item["created_at"][:10]
        if item_date != current_date:
            current_date = item_date
            lines.append(f"\n  [{current_date}]")
        src = f" [source: {item['source']}]" if item.get("source") else ""
        lines.append(f"    {item['entity_name']} ({item['entity_type']}): "
                     f"{item['content']}{src}")
        lines.append(f"      obs: {item['observation_id']}  entity: {item['entity_id']}")

    return "\n".join(lines)


def tool_point_in_time(entity_name_or_id: str, as_of: str,
                       vault: str = "", output_format: str = "text") -> str:
    """Get what was known about an entity at a specific point in time.

    Reconstructs the entity's state by including only observations that
    existed at the given timestamp, respecting superseding chains — if an
    observation was superseded before as_of, the replacement is shown instead.

    Args:
        entity_name_or_id: Entity name or ID.
        as_of: ISO date/datetime — the point in time to query.
        vault: Vault name (helps disambiguate).
        output_format: "text" (default) or "json".
    """
    output_format = (output_format or "text").lower()
    if output_format not in {"text", "json"}:
        return "Error: output_format must be 'text' or 'json'."

    as_of_dt = _parse_iso(as_of)
    if as_of_dt is None:
        return f"Error: invalid date '{as_of}'. Use ISO format (YYYY-MM-DD or full ISO datetime)."

    entity = resolve_entity(entity_name_or_id, vault or None)
    if entity is None:
        return f"Entity not found: '{entity_name_or_id}'"

    _load_store()

    # Get ALL observations for this entity (including superseded)
    all_obs = [
        o for o in _observations.values()
        if o.entity_id == entity.id and not o.deleted
    ]

    # Filter to observations that existed at as_of
    obs_at_time = []
    for obs in all_obs:
        obs_dt = _obs_created_dt(obs)
        if obs_dt is None:
            continue
        if obs_dt > as_of_dt:
            continue  # didn't exist yet

        # Check if this was superseded BEFORE as_of
        if obs.superseded_by:
            replacement = _observations.get(obs.superseded_by)
            if replacement:
                repl_dt = _obs_created_dt(replacement)
                if repl_dt and repl_dt <= as_of_dt:
                    continue  # was already superseded by as_of, skip

        obs_at_time.append(obs)

    obs_at_time.sort(key=lambda o: o.created_at)

    if output_format == "json":
        return json.dumps({
            "entity_id": entity.id,
            "entity_name": entity.name,
            "entity_type": entity.entity_type,
            "vault": entity.vault,
            "as_of": as_of,
            "observations": [
                {
                    "id": o.id,
                    "content": o.content,
                    "source": o.source,
                    "created_at": o.created_at,
                }
                for o in obs_at_time
            ],
        }, indent=2)

    if not obs_at_time:
        return f"No observations existed for '{entity.name}' as of {as_of}."

    lines = [
        f"Point-in-time snapshot: {entity.name} ({entity.entity_type}) as of {as_of}",
        f"  Entity ID: {entity.id}",
        f"  Vault: {entity.vault}",
        f"  Observations ({len(obs_at_time)}):",
    ]
    for obs in obs_at_time:
        src = f" [source: {obs.source}]" if obs.source else ""
        lines.append(f"    - {obs.content}{src}")
        lines.append(f"      created: {obs.created_at}  id: {obs.id}")

    return "\n".join(lines)


def tool_get_temporal_neighbors(entity_name_or_id: str, vault: str = "",
                                 direction: str = "both",
                                 limit: int = 10,
                                 output_format: str = "text") -> str:
    """Get entities temporally adjacent to a given entity.

    Finds graph neighbors and sorts them by their most recent observation
    timestamp, showing what was learned before or after the target entity.

    Args:
        entity_name_or_id: Entity name or ID.
        vault: Vault name (helps disambiguate).
        direction: "before", "after", or "both" (default).
        limit: Max results (default 10, max 50).
        output_format: "text" (default) or "json".
    """
    output_format = (output_format or "text").lower()
    if output_format not in {"text", "json"}:
        return "Error: output_format must be 'text' or 'json'."

    direction = (direction or "both").lower()
    if direction not in {"before", "after", "both"}:
        return "Error: direction must be 'before', 'after', or 'both'."

    limit = min(max(limit, 1), 50)

    entity = resolve_entity(entity_name_or_id, vault or None)
    if entity is None:
        return f"Entity not found: '{entity_name_or_id}'"

    _load_store()

    # Get the entity's temporal anchor (earliest observation)
    entity_obs = get_observations(entity.id)
    if not entity_obs:
        return f"No observations for '{entity.name}' — cannot determine temporal position."

    entity_times = [_obs_created_dt(o) for o in entity_obs]
    entity_times = [t for t in entity_times if t is not None]
    if not entity_times:
        return f"No valid timestamps for '{entity.name}'."

    entity_earliest = min(entity_times)

    # Get graph neighbors (2 hops to find more context)
    neighbors = get_neighbors(entity.id, max_depth=2)
    if not neighbors:
        return f"No graph neighbors for '{entity.name}'."

    # For each neighbor, get their temporal info
    temporal_neighbors = []
    for nb in neighbors:
        nb_ent = get_entity(nb["entity_id"])
        if nb_ent is None:
            continue

        nb_obs = get_observations(nb_ent.id)
        if not nb_obs:
            continue

        nb_times = [_obs_created_dt(o) for o in nb_obs]
        nb_times = [t for t in nb_times if t is not None]
        if not nb_times:
            continue

        nb_earliest = min(nb_times)
        nb_latest = max(nb_times)

        # Filter by direction
        if direction == "before" and nb_latest >= entity_earliest:
            continue
        if direction == "after" and nb_earliest <= entity_earliest:
            continue

        temporal_neighbors.append({
            "entity_id": nb_ent.id,
            "entity_name": nb_ent.name,
            "entity_type": nb_ent.entity_type,
            "vault": nb_ent.vault,
            "earliest_observation": min(nb_times).isoformat(),
            "latest_observation": max(nb_times).isoformat(),
            "observation_count": len(nb_obs),
            "relation_type": nb["relation_type"],
            "relation_direction": nb["direction"],
            "depth": nb["depth"],
        })

    # Sort by earliest observation
    temporal_neighbors.sort(key=lambda x: x["earliest_observation"])
    temporal_neighbors = temporal_neighbors[:limit]

    if not temporal_neighbors:
        return f"No temporal neighbors found for '{entity.name}' (direction: {direction})."

    if output_format == "json":
        return json.dumps({
            "entity_id": entity.id,
            "entity_name": entity.name,
            "anchor_time": entity_earliest.isoformat(),
            "direction": direction,
            "neighbors": temporal_neighbors,
        }, indent=2)

    # Text format
    lines = [
        f"Temporal neighbors of '{entity.name}' (anchor: {entity_earliest.isoformat()[:10]}, "
        f"direction: {direction}):",
    ]
    for tn in temporal_neighbors:
        arrow = "->" if tn["relation_direction"] == "outgoing" else "<-"
        lines.append(
            f"  {arrow} {tn['entity_name']} ({tn['entity_type']}) "
            f"[{tn['relation_type']}]"
        )
        lines.append(
            f"     {tn['earliest_observation'][:10]} — {tn['latest_observation'][:10]} "
            f"({tn['observation_count']} observations)"
        )

    return "\n".join(lines)
