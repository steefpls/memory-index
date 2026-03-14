"""Graph analysis tool — PageRank, community detection, knowledge gaps."""

import json
import logging

from src.graph.traversal import compute_pagerank, detect_communities, find_knowledge_gaps
from src.indexer.store import get_entity, get_observations

logger = logging.getLogger(__name__)


def tool_analyze_graph(vault: str = "", top_n: int = 20,
                       output_format: str = "text") -> str:
    """Analyze the knowledge graph: PageRank, communities, knowledge gaps.

    Args:
        vault: Optional vault filter for knowledge gaps (empty = all).
        top_n: Number of top PageRank results (default 20).
        output_format: "text" (default) or "json".

    Returns:
        Analysis results.
    """
    output_format = (output_format or "text").lower()
    if output_format not in {"text", "json"}:
        return "Error: output_format must be 'text' or 'json'."

    top_n = min(max(top_n, 1), 50)

    # PageRank
    pr_results = compute_pagerank(top_n=top_n)
    # Hydrate with entity names
    for item in pr_results:
        ent = get_entity(item["entity_id"])
        if ent:
            item["entity_name"] = ent.name
            item["entity_type"] = ent.entity_type
            item["vault"] = ent.vault

    # Communities
    communities = detect_communities()
    # Hydrate community members
    hydrated_communities = []
    for community in communities[:10]:  # cap at 10 communities
        members = []
        for eid in community:
            ent = get_entity(eid)
            if ent:
                members.append({
                    "entity_id": eid,
                    "entity_name": ent.name,
                    "entity_type": ent.entity_type,
                })
            else:
                members.append({"entity_id": eid, "entity_name": "?", "entity_type": "?"})
        hydrated_communities.append(members)

    # Knowledge gaps
    gaps = find_knowledge_gaps(min_observations=2)
    # Filter by vault if specified
    if vault:
        gaps = [g for g in gaps if g.get("vault") == vault]
    gaps = gaps[:20]  # cap

    if output_format == "json":
        return json.dumps({
            "pagerank": pr_results,
            "communities": hydrated_communities,
            "knowledge_gaps": gaps,
        }, indent=2)

    # Text format
    lines = []

    # PageRank section
    lines.append(f"PageRank — Top {len(pr_results)} entities by centrality:")
    for i, item in enumerate(pr_results, 1):
        name = item.get("entity_name", item["entity_id"])
        etype = item.get("entity_type", "?")
        lines.append(f"  {i}. {name} ({etype}) — PR: {item['pagerank']:.6f}")
    lines.append("")

    # Communities section
    lines.append(f"Communities — {len(hydrated_communities)} detected (Louvain):")
    for i, members in enumerate(hydrated_communities, 1):
        member_names = [m["entity_name"] for m in members]
        lines.append(f"  Community {i} ({len(members)} members):")
        # Show up to 8 members per community
        shown = member_names[:8]
        if len(member_names) > 8:
            shown.append(f"...+{len(member_names) - 8} more")
        lines.append(f"    {', '.join(shown)}")
    lines.append("")

    # Knowledge gaps section
    if gaps:
        lines.append(f"Knowledge gaps — {len(gaps)} under-documented entities (high PageRank, <2 observations):")
        for item in gaps:
            lines.append(
                f"  {item['entity_name']} ({item['entity_type']}) — "
                f"PR: {item['pagerank']:.6f}, observations: {item['observation_count']}"
            )
    else:
        lines.append("Knowledge gaps: none found (all central entities are well-documented).")

    return "\n".join(lines)
