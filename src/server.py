"""Memory Index MCP server — persistent entity/observation/relation memory.

Provides 22 MCP tools for knowledge management via FastMCP:
- 5 entity tools, 2 observation tools, 2 relation tools
- 2 search tools (semantic + spreading activation with RRF fusion)
- 3 temporal tools (timeline, point-in-time, temporal neighbors)
- 3 graph analysis tools (PageRank/Louvain/gaps + Librarian clustering + visualization)
- 5 status/vault tools
"""

import logging
import os
import sys
import threading
from pathlib import Path

# Ensure project root is on sys.path when run as `python src/server.py`
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

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
                    supersedes: str = "") -> str:
    """Add an observation (fact) to an existing entity.

    Args:
        name_or_id: Entity name or ID.
        content: The observation text.
        vault: Vault name (helps disambiguate names).
        source: Optional source attribution.
        supersedes: Optional observation ID that this replaces. The old
                    observation is kept for history but excluded from search.
    """
    from src.tools.entities import tool_add_observation
    return tool_add_observation(name_or_id, content, vault, source, supersedes)


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
                   entity_type: str = "",
                   since: str = "", before: str = "",
                   include_superseded: bool = False,
                   strategy: str = "associative",
                   output_format: str = "text") -> str:
    """Semantic memory search with spreading activation and RRF fusion.

    Finds entities and observations matching your query using vector similarity,
    then explores the knowledge graph outward from hits via spreading activation,
    and merges both rankings with Reciprocal Rank Fusion.

    Args:
        query: Natural language query describing what you're looking for.
        vault: Vault to search (empty = search all vaults).
        n_results: Number of results (default 10, max 30).
        entity_type: Optional entity type filter.
        since: Only include observations created after this ISO date/datetime
               (e.g., "2026-03-01", "2026-03-13T10:00:00").
        before: Only include observations created before this ISO date/datetime.
        include_superseded: Include observations that have been replaced by
                            newer ones (default False). Useful for history queries.
        strategy: "associative" (default — spreading activation + RRF fusion)
                  or "semantic" (vector-only, no graph expansion).
        output_format: "text" (default) or "json".
    """
    from src.tools.search import search_memory as do_search
    return do_search(query, vault, n_results, entity_type,
                     since, before, include_superseded, strategy, output_format)


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


# ========== Temporal Tools ==========

@mcp.tool()
def query_timeline(vault: str = "", start: str = "", end: str = "",
                   entity_type: str = "", limit: int = 50,
                   output_format: str = "text") -> str:
    """Query observations across a time range, ordered chronologically.

    Returns a timeline of facts/events within the specified window,
    grouped by date.

    Args:
        vault: Vault to query (empty = all vaults).
        start: Start date/datetime (ISO format, inclusive). Empty = no lower bound.
        end: End date/datetime (ISO format, exclusive). Empty = no upper bound.
        entity_type: Optional filter by entity type.
        limit: Max results (default 50, max 200).
        output_format: "text" (default) or "json".
    """
    from src.tools.temporal import tool_query_timeline
    return tool_query_timeline(vault, start, end, entity_type, limit, output_format)


@mcp.tool()
def point_in_time(entity_name_or_id: str, as_of: str,
                  vault: str = "", output_format: str = "text") -> str:
    """Get what was known about an entity at a specific point in time.

    Reconstructs entity state by including only observations that existed
    at the given timestamp, correctly handling superseding chains — if an
    observation was superseded before as_of, only the replacement is shown.

    Args:
        entity_name_or_id: Entity name or ID.
        as_of: ISO date/datetime — the point in time to query.
        vault: Vault name (helps disambiguate).
        output_format: "text" (default) or "json".
    """
    from src.tools.temporal import tool_point_in_time
    return tool_point_in_time(entity_name_or_id, as_of, vault, output_format)


@mcp.tool()
def get_temporal_neighbors(entity_name_or_id: str, vault: str = "",
                           direction: str = "both", limit: int = 10,
                           output_format: str = "text") -> str:
    """Get entities temporally adjacent to a given entity.

    Finds graph neighbors and sorts them by observation timestamps,
    showing what was learned before or after the target entity.

    Args:
        entity_name_or_id: Entity name or ID.
        vault: Vault name (helps disambiguate).
        direction: "before", "after", or "both" (default).
        limit: Max results (default 10, max 50).
        output_format: "text" (default) or "json".
    """
    from src.tools.temporal import tool_get_temporal_neighbors
    return tool_get_temporal_neighbors(entity_name_or_id, vault, direction, limit, output_format)


# ========== Graph Analysis Tools ==========

@mcp.tool()
def analyze_graph(vault: str = "", top_n: int = 20,
                  output_format: str = "text") -> str:
    """Analyze the knowledge graph: PageRank centrality, community detection,
    and knowledge gap identification.

    Returns the most important entities (by PageRank), detected communities
    (Louvain), and under-documented entities that may need more observations.

    Args:
        vault: Optional vault filter for knowledge gaps (empty = all).
        top_n: Number of top PageRank results (default 20).
        output_format: "text" (default) or "json".
    """
    from src.tools.graph_analysis import tool_analyze_graph
    return tool_analyze_graph(vault, top_n, output_format)


@mcp.tool()
def run_librarian(vault: str = "", eps: float = 0.5,
                  min_samples: int = 2,
                  output_format: str = "text") -> str:
    """Run the Librarian: discover knowledge clusters and structural gaps.

    Analyzes all observation embeddings using DBSCAN clustering to find
    concept groups, then checks the knowledge graph for structural gaps —
    clusters of semantically similar entities that lack graph relations.
    Produces an actionable report suggesting missing connections.

    Args:
        vault: Vault to analyze (required).
        eps: DBSCAN epsilon — max cosine distance within a cluster.
             Lower = tighter clusters (default 0.5).
        min_samples: Min observations to form a cluster (default 2).
        output_format: "text" (default) or "json".
    """
    from src.tools.librarian import tool_run_librarian
    return tool_run_librarian(vault, eps, min_samples, output_format)


@mcp.tool()
def visualize_graph(vault: str = "") -> str:
    """Generate an interactive knowledge graph visualization and open in browser.

    Builds a self-contained HTML file with Cytoscape.js showing all entities
    as color-coded nodes and relations as directed edges. Click a node to see
    its observations and relations in a side panel.

    Args:
        vault: Optional vault filter (empty = all vaults).
    """
    from src.tools.visualize import tool_visualize_graph
    return tool_visualize_graph(vault)


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


@mcp.tool()
def delete_vault(name: str) -> str:
    """Delete a vault and all its entities, observations, relations, and vectors.

    Args:
        name: Vault name to delete.
    """
    from src.tools.status import tool_delete_vault
    return tool_delete_vault(name)


# ========== Import / Export ==========

@mcp.tool()
def export_vault(vault: str, output_path: str = "") -> str:
    """Export a vault to a portable zip archive.

    The archive contains JSON files with entities, observations, and
    relations. Vector embeddings are not included — they are regenerated
    on import, which keeps the archive small and portable across embedding
    model changes.

    Args:
        vault: Vault name to export.
        output_path: Optional output path. May be a directory (filename
                     auto-generated as <vault>_<timestamp>.zip), an explicit
                     .zip path, or empty (defaults to data/exports/).
    """
    from src.tools.portability import tool_export_vault
    return tool_export_vault(vault, output_path)


@mcp.tool()
def vacuum_store(dry_run: bool = False) -> str:
    """Hard-remove stale rows that accumulated despite soft-delete semantics.

    Removes entities whose vault no longer exists, soft-deleted entities,
    observations whose entity is gone, and relations with dangling endpoints.
    Affected Chroma vectors are removed best-effort.

    Args:
        dry_run: If True, report what would be removed without modifying state.
    """
    from src.tools.maintenance import tool_vacuum_store
    return tool_vacuum_store(dry_run)


@mcp.tool()
def import_vault(input_path: str, vault: str = "") -> str:
    """Import a vault export zip into a target vault. Always additive.

    Existing entities are matched by name and reused. Observations are
    deduped by exact content per entity. Relations are deduped by
    (from, to, relation_type). Nothing in the target vault is removed.

    Args:
        input_path: Path to a zip archive produced by export_vault.
        vault: Target vault name. Defaults to the source vault recorded in
               the archive's manifest. Auto-created if it doesn't exist.
    """
    from src.tools.portability import tool_import_vault
    return tool_import_vault(input_path, vault)


# --- Startup ---

def _startup_check():
    """Validate state and start search init in background."""
    from src.config import VAULTS
    logger.info("memory-index starting: %d vaults configured", len(VAULTS))

    # Pre-import heavy C extension libraries (numpy, sklearn, onnxruntime)
    # BEFORE launching the background init thread. On Windows, concurrent
    # first-imports of C extensions can deadlock on the DLL loader lock.
    import numpy  # noqa: F401
    try:
        import sklearn  # noqa: F401
    except ImportError:
        pass

    # Start search backend init eagerly
    def _bg_init():
        try:
            from src.tools.search import start_search_init
            start_search_init()
        except Exception as e:
            logger.warning("Background search init failed: %s", e)

    threading.Thread(target=_bg_init, daemon=True, name="memory-index-startup").start()


if __name__ == "__main__":
    _startup_check()
    mcp.run(transport="stdio")
