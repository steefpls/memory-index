"""Memory Index MCP server — persistent entity/observation/relation memory.

Provides 15 MCP tools for knowledge management via FastMCP.
"""

import logging
import os
import sys
import threading

from mcp.server.fastmcp import FastMCP

from src.config import DATA_DIR, LOG_FILE, PROJECT_ROOT

# --- Logging setup ---
DATA_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(str(LOG_FILE), encoding="utf-8"),
        logging.StreamHandler(sys.stderr),
    ],
)
logger = logging.getLogger(__name__)

# Disable stdout (stdio transport only)
sys.stdout = open(os.devnull, "w")

# --- FastMCP server ---
mcp = FastMCP("memory-index")


# ========== Entity Tools ==========

@mcp.tool()
def create_entity(name: str, entity_type: str, vault: str,
                  observations: str = "", source: str = "") -> str:
    """Create a named entity with optional initial observations.

    Args:
        name: Entity name (e.g., "Python", "memory-index", "Alice").
        entity_type: Type (person, project, concept, decision, error,
                     solution, technology, event, organization, etc.).
        vault: Vault to store in. Created automatically if it doesn't exist.
        observations: Pipe-separated observations (e.g., "Fact 1|Fact 2|Fact 3").
        source: Optional source attribution for observations.
    """
    from src.tools.entities import tool_create_entity
    return tool_create_entity(name, entity_type, vault, observations, source)


@mcp.tool()
def get_entity(name_or_id: str, vault: str = "") -> str:
    """Get entity details including observations and relations.

    Args:
        name_or_id: Entity name or ID.
        vault: Vault name (helps disambiguate names across vaults).
    """
    from src.tools.entities import tool_get_entity
    return tool_get_entity(name_or_id, vault)


@mcp.tool()
def update_entity(name_or_id: str, new_name: str = "",
                  new_type: str = "", vault: str = "") -> str:
    """Update an entity's name or type.

    Args:
        name_or_id: Current entity name or ID.
        new_name: New name (leave empty to keep current).
        new_type: New type (leave empty to keep current).
        vault: Vault name (helps disambiguate names).
    """
    from src.tools.entities import tool_update_entity
    return tool_update_entity(name_or_id, new_name, new_type, vault)


@mcp.tool()
def delete_entity(name_or_id: str, vault: str = "") -> str:
    """Soft delete an entity and its observations.

    Args:
        name_or_id: Entity name or ID.
        vault: Vault name (helps disambiguate names).
    """
    from src.tools.entities import tool_delete_entity
    return tool_delete_entity(name_or_id, vault)


@mcp.tool()
def list_entities(vault: str = "", entity_type: str = "",
                  offset: int = 0, limit: int = 20) -> str:
    """List entities with optional filters.

    Args:
        vault: Filter by vault (empty = all vaults).
        entity_type: Filter by type (empty = all types).
        offset: Pagination offset.
        limit: Max results (1-50).
    """
    from src.tools.entities import tool_list_entities
    return tool_list_entities(vault, entity_type, offset, limit)


# ========== Observation Tools ==========

@mcp.tool()
def add_observation(name_or_id: str, content: str,
                    vault: str = "", source: str = "",
                    confidence: float = 1.0) -> str:
    """Add an observation (fact) to an existing entity.

    Args:
        name_or_id: Entity name or ID.
        content: The observation text.
        vault: Vault name (helps disambiguate names).
        source: Optional source attribution.
        confidence: Confidence level (0.0 to 1.0, default 1.0).
    """
    from src.tools.entities import tool_add_observation
    return tool_add_observation(name_or_id, content, vault, source, confidence)


@mcp.tool()
def delete_observation(observation_id: str) -> str:
    """Remove an observation by ID.

    Args:
        observation_id: The observation ID to delete.
    """
    from src.tools.entities import tool_delete_observation
    return tool_delete_observation(observation_id)


# ========== Relation Tools ==========

@mcp.tool()
def create_relation(from_entity: str, to_entity: str,
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
    """
    from src.tools.relations import tool_create_relation
    return tool_create_relation(from_entity, to_entity, relation_type, vault, weight, context)


@mcp.tool()
def delete_relation(relation_id: str) -> str:
    """Remove a relation by ID.

    Args:
        relation_id: The relation ID to delete.
    """
    from src.tools.relations import tool_delete_relation
    return tool_delete_relation(relation_id)


# ========== Search Tools ==========

@mcp.tool()
def search_memory(query: str, vault: str = "", n_results: int = 10,
                   entity_type: str = "", include_neighbors: bool = True,
                   output_format: str = "text") -> str:
    """Semantic memory search with graph boosting across knowledge entities.

    Finds entities and observations matching your query using vector similarity,
    then expands results via graph relations for connected context.

    Args:
        query: Natural language query describing what you're looking for.
        vault: Vault to search (empty = search all vaults).
        n_results: Number of results (default 10, max 30).
        entity_type: Optional entity type filter.
        include_neighbors: Include graph-connected entities (default True).
        output_format: "text" (default) or "json".
    """
    from src.tools.search import search_memory as do_search
    return do_search(query, vault, n_results, entity_type, include_neighbors, output_format)


@mcp.tool()
def get_neighbors(entity_name_or_id: str, vault: str = "",
                  max_depth: int = 1, relation_type: str = "") -> str:
    """Get entities connected to a given entity via graph relations.

    Args:
        entity_name_or_id: Entity name or ID.
        vault: Vault name (helps disambiguate names).
        max_depth: How many hops to traverse (default 1, max 3).
        relation_type: Optional filter by relation type.
    """
    from src.indexer.store import resolve_entity
    from src.graph.traversal import get_neighbors as graph_neighbors
    from src.indexer.store import get_entity as store_get_entity

    entity = resolve_entity(entity_name_or_id, vault or None)
    if entity is None:
        return f"Entity not found: '{entity_name_or_id}'"

    max_depth = min(max(max_depth, 1), 3)
    neighbors = graph_neighbors(
        entity.id, max_depth=max_depth,
        relation_type=relation_type or None,
    )

    if not neighbors:
        return f"No neighbors found for '{entity.name}'."

    lines = [f"Neighbors of '{entity.name}' ({len(neighbors)} found):"]
    for nb in neighbors:
        nb_ent = store_get_entity(nb["entity_id"])
        nb_name = nb_ent.name if nb_ent else nb["entity_id"]
        nb_type = f" ({nb_ent.entity_type})" if nb_ent else ""
        direction = "->" if nb["direction"] == "outgoing" else "<-"
        ctx = f" — {nb['context']}" if nb.get("context") else ""
        lines.append(f"  {direction} {nb_name}{nb_type} [{nb['relation_type']}] "
                     f"(depth: {nb['depth']}, weight: {nb['weight']}){ctx}")

    return "\n".join(lines)


# ========== Status Tools ==========

@mcp.tool()
def get_graph_summary() -> str:
    """Get knowledge graph statistics: node/edge counts, components, relation types."""
    from src.tools.status import tool_get_graph_summary
    return tool_get_graph_summary()


@mcp.tool()
def memory_status() -> str:
    """Health check showing backend status, entity/observation/relation counts per vault."""
    from src.tools.status import tool_memory_status
    return tool_memory_status()


@mcp.tool()
def list_vaults() -> str:
    """List all memory vaults with entity and observation counts."""
    from src.tools.status import tool_list_vaults
    return tool_list_vaults()


@mcp.tool()
def create_vault(name: str) -> str:
    """Create a new memory vault for isolating knowledge domains.

    Args:
        name: Vault name (alphanumeric + underscores recommended).
    """
    from src.tools.status import tool_create_vault
    return tool_create_vault(name)


# --- Startup ---

def _startup_check():
    """Validate state and start search init in background."""
    from src.config import VAULTS
    logger.info("memory-index starting: %d vaults configured", len(VAULTS))

    # Start search backend init eagerly
    def _bg_init():
        try:
            from src.tools.search import start_search_init
            start_search_init()
        except Exception as e:
            logger.warning("Background search init failed: %s", e)

    threading.Thread(target=_bg_init, daemon=True, name="memory-index-startup").start()


_startup_check()


if __name__ == "__main__":
    mcp.run(transport="stdio")
