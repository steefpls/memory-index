"""Health check and vault management tool implementations."""

import logging

from src.config import (
    get_vault, list_vaults as config_list_vaults,
    create_vault as config_create_vault, delete_vault as config_delete_vault,
)
from src.indexer.store import get_entity_count, get_observation_count
from src.indexer.embedder import get_active_backend
from src.graph.manager import get_relation_count
from src.graph.traversal import get_graph_summary

logger = logging.getLogger(__name__)


def tool_memory_status() -> str:
    """Health check showing backend status, entity/observation/relation counts.

    Returns:
        Status summary.
    """
    backend = get_active_backend(role="search")

    vaults = config_list_vaults()
    vault_details = []
    total_entities = 0
    total_observations = 0

    for vault in vaults:
        e_count = get_entity_count(vault.name)
        o_count = get_observation_count(vault.name)
        total_entities += e_count
        total_observations += o_count
        vault_details.append(f"  {vault.name}: {e_count} entities, {o_count} observations")

    rel_count = get_relation_count()
    graph_summary = get_graph_summary()

    lines = [
        "Memory Index Status",
        f"  Backend: {backend}",
        f"  Vaults: {len(vaults)}",
        f"  Total entities: {total_entities}",
        f"  Total observations: {total_observations}",
        f"  Total relations: {rel_count}",
        f"  Graph: {graph_summary['nodes']} nodes, {graph_summary['edges']} edges, "
        f"{graph_summary['components']} components",
    ]

    if vault_details:
        lines.append("")
        lines.append("Vault details:")
        lines.extend(vault_details)

    if graph_summary["relation_types"]:
        lines.append("")
        lines.append("Relation types:")
        for rt, count in sorted(graph_summary["relation_types"].items()):
            lines.append(f"  {rt}: {count}")

    return "\n".join(lines)


def tool_list_vaults() -> str:
    """List all memory vaults.

    Returns:
        Vault list with details.
    """
    vaults = config_list_vaults()
    if not vaults:
        return "No vaults configured. Use create_vault() to create one."

    lines = [f"Vaults ({len(vaults)}):"]
    for vault in vaults:
        e_count = get_entity_count(vault.name)
        o_count = get_observation_count(vault.name)
        lines.append(f"  {vault.name} (collection: {vault.collection_name})")
        lines.append(f"    {e_count} entities, {o_count} observations")
        if vault.created_at:
            lines.append(f"    Created: {vault.created_at}")

    return "\n".join(lines)


def tool_create_vault(name: str) -> str:
    """Create a new memory vault.

    Args:
        name: Vault name (alphanumeric + underscores recommended).

    Returns:
        Confirmation or error.
    """
    if not name or not name.strip():
        return "Error: vault name is required."

    name = name.strip()
    if get_vault(name) is not None:
        return f"Vault '{name}' already exists."

    vault = config_create_vault(name)
    return f"Vault created: {vault.name} (collection: {vault.collection_name})"


def tool_delete_vault(name: str) -> str:
    """Delete a vault and all its entities, observations, and relations.

    Args:
        name: Vault name to delete.

    Returns:
        Confirmation or error.
    """
    if not name or not name.strip():
        return "Error: vault name is required."

    name = name.strip()
    vault_cfg = get_vault(name)
    if vault_cfg is None:
        return f"Vault '{name}' not found."

    from src.indexer import store as store_mod
    from src.graph import manager as graph_mod
    from src.indexer.embedder import get_chroma_client
    from src.config import DATA_DIR

    # Hard-remove entities, observations, and relations for this vault.
    # We can drop them outright (rather than soft-delete) because the entire
    # vault is going away — there's no audit trail to preserve.
    store_mod._load_store()
    graph_mod._get_graph()

    vault_entity_ids = {
        e.id for e in store_mod._entities.values() if e.vault == name
    }
    entity_count = len(vault_entity_ids)

    # Remove relations involving any of these entities
    for eid in vault_entity_ids:
        graph_mod.remove_entity_relations(eid)

    # Hard-pop observations and entities from the in-memory stores
    obs_ids_to_drop = [
        oid for oid, o in store_mod._observations.items()
        if o.entity_id in vault_entity_ids
    ]
    for oid in obs_ids_to_drop:
        store_mod._observations.pop(oid, None)
    for eid in vault_entity_ids:
        store_mod._entities.pop(eid, None)

    store_mod._save_store()

    # Drop the ChromaDB collection (cleans all vectors at once)
    try:
        client = get_chroma_client()
        client.delete_collection(vault_cfg.collection_name)
    except Exception as e:
        logger.warning("Failed to delete ChromaDB collection: %s", e)

    # Remove the per-vault calibration sidecar, if present
    cal_path = DATA_DIR / f"{name}_calibration.json"
    if cal_path.exists():
        try:
            cal_path.unlink()
        except OSError as e:
            logger.warning("Failed to remove calibration file %s: %s", cal_path, e)

    # Remove vault config
    config_delete_vault(name)

    return f"Vault '{name}' deleted ({entity_count} entities removed)."


def tool_get_graph_summary() -> str:
    """Get knowledge graph statistics.

    Returns:
        Graph summary with node/edge counts, components, and relation type distribution.
    """
    summary = get_graph_summary()

    lines = [
        "Knowledge Graph Summary",
        f"  Nodes: {summary['nodes']}",
        f"  Edges: {summary['edges']}",
        f"  Connected components: {summary['components']}",
    ]

    if summary["relation_types"]:
        lines.append("")
        lines.append("Relation type distribution:")
        for rt, count in sorted(summary["relation_types"].items()):
            lines.append(f"  {rt}: {count}")
    else:
        lines.append("  No relations yet.")

    return "\n".join(lines)
