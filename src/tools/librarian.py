"""The Librarian — autonomous knowledge graph analysis and organization.

Runs DBSCAN clustering on observation embeddings to discover concept groups,
detects structural gaps between semantically similar but poorly connected
clusters, and produces an actionable report.
"""

import json
import logging
from datetime import datetime, timezone

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances

from src.config import VAULTS, get_vault
from src.indexer.embedder import get_collection
from src.indexer.store import get_entity, get_observations, _load_store, _entities
from src.graph.traversal import get_graph_summary
from src.graph.manager import get_graph

logger = logging.getLogger(__name__)


def _fetch_embeddings(vault_name: str) -> tuple[list[str], list[str], np.ndarray]:
    """Fetch all observation embeddings from a vault's ChromaDB collection.

    Returns:
        (observation_ids, entity_ids, embeddings_matrix)
    """
    vault_cfg = get_vault(vault_name)
    if vault_cfg is None:
        return [], [], np.array([])

    collection = get_collection(vault_cfg.collection_name)
    count = collection.count()
    if count == 0:
        return [], [], np.array([])

    # Fetch all in batches (ChromaDB limits)
    all_ids = []
    all_entity_ids = []
    all_embeddings = []

    batch_size = 5000
    offset = 0
    while offset < count:
        result = collection.get(
            include=["embeddings", "metadatas"],
            limit=batch_size,
            offset=offset,
        )
        if not result["ids"]:
            break

        all_ids.extend(result["ids"])
        for meta in result["metadatas"]:
            all_entity_ids.append(meta.get("entity_id", ""))
        all_embeddings.extend(result["embeddings"])

        offset += len(result["ids"])
        if len(result["ids"]) < batch_size:
            break

    if not all_embeddings:
        return [], [], np.array([])

    return all_ids, all_entity_ids, np.array(all_embeddings, dtype=np.float32)


def _run_dbscan(embeddings: np.ndarray, eps: float = 0.5,
                min_samples: int = 2) -> np.ndarray:
    """Run DBSCAN clustering on embeddings using cosine distance.

    Args:
        embeddings: (N, D) matrix of observation embeddings.
        eps: Maximum distance between two samples in a cluster.
        min_samples: Minimum samples to form a cluster.

    Returns:
        Cluster labels array (length N). -1 = noise.
    """
    # Cosine distance matrix
    dist_matrix = cosine_distances(embeddings)
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
    return clustering.fit_predict(dist_matrix)


def _build_clusters(labels: np.ndarray, obs_ids: list[str],
                    entity_ids: list[str],
                    embeddings: np.ndarray) -> list[dict]:
    """Build cluster metadata from DBSCAN labels.

    Returns:
        List of cluster dicts with id, centroid, cohesion, entity composition.
    """
    _load_store()

    unique_labels = set(labels)
    unique_labels.discard(-1)  # remove noise

    clusters = []
    for label in sorted(unique_labels):
        mask = labels == label
        indices = np.where(mask)[0]
        cluster_embeddings = embeddings[indices]

        # Centroid and cohesion
        centroid = cluster_embeddings.mean(axis=0)
        if len(cluster_embeddings) > 1:
            dists_to_centroid = cosine_distances(cluster_embeddings, centroid.reshape(1, -1)).flatten()
            cohesion = float(dists_to_centroid.mean())
        else:
            cohesion = 0.0

        # Entity composition
        cluster_entity_ids = set()
        for idx in indices:
            eid = entity_ids[idx]
            if eid:
                cluster_entity_ids.add(eid)

        # Hydrate entity names
        entity_info = []
        for eid in cluster_entity_ids:
            ent = get_entity(eid)
            if ent:
                entity_info.append({
                    "entity_id": eid,
                    "entity_name": ent.name,
                    "entity_type": ent.entity_type,
                })

        clusters.append({
            "cluster_id": int(label),
            "size": int(mask.sum()),
            "cohesion": round(cohesion, 4),
            "entities": entity_info,
            "centroid": centroid,  # kept for gap detection, stripped from output
        })

    return clusters


def _detect_gaps(clusters: list[dict], graph) -> list[dict]:
    """Detect structural gaps: clusters that are semantically similar but
    poorly connected in the knowledge graph.

    A gap means "these concepts are related in meaning but you haven't
    linked them with relations."
    """
    if len(clusters) < 2:
        return []

    # Compute pairwise cosine similarity between cluster centroids
    centroids = np.array([c["centroid"] for c in clusters])
    dist_matrix = cosine_distances(centroids)

    gaps = []
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            similarity = 1.0 - dist_matrix[i][j]

            # Only consider clusters with high semantic similarity
            if similarity < 0.6:
                continue

            # Count cross-cluster edges in the graph
            ids_a = {e["entity_id"] for e in clusters[i]["entities"]}
            ids_b = {e["entity_id"] for e in clusters[j]["entities"]}

            cross_edges = 0
            for eid_a in ids_a:
                if not graph.has_node(eid_a):
                    continue
                for _, target, _ in graph.out_edges(eid_a, data=True):
                    if target in ids_b:
                        cross_edges += 1
                for source, _, _ in graph.in_edges(eid_a, data=True):
                    if source in ids_b:
                        cross_edges += 1

            # Gap = high similarity but few/no edges
            if cross_edges < 2:
                # Suggest bridge entities (entities from each cluster that
                # might be worth connecting)
                suggest_a = clusters[i]["entities"][:3]
                suggest_b = clusters[j]["entities"][:3]

                gaps.append({
                    "cluster_a": clusters[i]["cluster_id"],
                    "cluster_b": clusters[j]["cluster_id"],
                    "similarity": round(similarity, 4),
                    "cross_edges": cross_edges,
                    "suggest_from": [e["entity_name"] for e in suggest_a],
                    "suggest_to": [e["entity_name"] for e in suggest_b],
                })

    # Sort by similarity descending (most obvious gaps first)
    gaps.sort(key=lambda x: -x["similarity"])
    return gaps


def tool_run_librarian(vault: str = "", eps: float = 0.5,
                       min_samples: int = 2,
                       output_format: str = "text") -> str:
    """Run the Librarian: discover knowledge clusters and structural gaps.

    Analyzes all observation embeddings in a vault using DBSCAN clustering
    to find concept groups, then checks the knowledge graph for structural
    gaps — clusters of semantically similar entities that lack graph relations.

    Args:
        vault: Vault to analyze (required).
        eps: DBSCAN epsilon — max cosine distance within a cluster.
             Lower = tighter clusters (default 0.5).
        min_samples: Min observations to form a cluster (default 2).
        output_format: "text" (default) or "json".
    """
    output_format = (output_format or "text").lower()
    if output_format not in {"text", "json"}:
        return "Error: output_format must be 'text' or 'json'."

    if not vault:
        return "Error: vault is required. Use list_vaults() to see available vaults."

    if vault not in VAULTS:
        return f"Error: Unknown vault '{vault}'. Use list_vaults() to see available vaults."

    # Fetch embeddings
    obs_ids, entity_ids, embeddings = _fetch_embeddings(vault)
    if len(obs_ids) == 0:
        return f"No observations in vault '{vault}'. Nothing to analyze."

    if len(obs_ids) < min_samples:
        return f"Only {len(obs_ids)} observations in vault '{vault}'. Need at least {min_samples} for clustering."

    # Run DBSCAN
    labels = _run_dbscan(embeddings, eps=eps, min_samples=min_samples)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())

    # Build cluster metadata
    clusters = _build_clusters(labels, obs_ids, entity_ids, embeddings)

    # Detect structural gaps
    graph = get_graph()
    gaps = _detect_gaps(clusters, graph)

    # Build report
    report = {
        "vault": vault,
        "total_observations": len(obs_ids),
        "clusters_found": n_clusters,
        "noise_observations": n_noise,
        "eps": eps,
        "min_samples": min_samples,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "clusters": [
            {
                "cluster_id": c["cluster_id"],
                "size": c["size"],
                "cohesion": c["cohesion"],
                "entities": c["entities"],
            }
            for c in clusters
        ],
        "structural_gaps": gaps,
    }

    if output_format == "json":
        return json.dumps(report, indent=2)

    return _format_report_text(report)


def _format_report_text(report: dict) -> str:
    """Format librarian report as readable text."""
    lines = [
        f"Librarian Report: vault '{report['vault']}'",
        f"  {report['total_observations']} observations, "
        f"{report['clusters_found']} clusters, "
        f"{report['noise_observations']} noise (unclustered)",
        f"  DBSCAN params: eps={report['eps']}, min_samples={report['min_samples']}",
        "",
    ]

    # Clusters
    if report["clusters"]:
        lines.append("Knowledge Clusters:")
        for cluster in report["clusters"]:
            entity_names = [e["entity_name"] for e in cluster["entities"]]
            shown = entity_names[:8]
            extra = f", ...+{len(entity_names) - 8}" if len(entity_names) > 8 else ""

            lines.append(
                f"  Cluster {cluster['cluster_id']} "
                f"({cluster['size']} observations, "
                f"cohesion: {cluster['cohesion']:.3f}):"
            )
            lines.append(f"    Entities: {', '.join(shown)}{extra}")
        lines.append("")

    # Gaps
    if report["structural_gaps"]:
        lines.append("Structural Gaps (semantically similar but poorly connected):")
        for gap in report["structural_gaps"]:
            lines.append(
                f"  Cluster {gap['cluster_a']} <-> Cluster {gap['cluster_b']} "
                f"(similarity: {gap['similarity']:.2f}, cross-edges: {gap['cross_edges']})"
            )
            lines.append(f"    Consider linking: {', '.join(gap['suggest_from'])} "
                         f"<-> {', '.join(gap['suggest_to'])}")
        lines.append("")
    else:
        lines.append("Structural Gaps: none found (all similar clusters are well-connected).")

    return "\n".join(lines)
