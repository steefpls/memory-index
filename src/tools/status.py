"""Health check and vault management tool implementations."""

import logging

from src.config import VAULTS, list_vaults as config_list_vaults, create_vault as config_create_vault
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
    if name in VAULTS:
        return f"Vault '{name}' already exists."

    vault = config_create_vault(name)
    return f"Vault created: {vault.name} (collection: {vault.collection_name})"


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
