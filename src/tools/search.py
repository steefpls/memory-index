"""Hybrid vector + graph-boosted memory search."""

import json
import logging
import os
import threading
import time

from src.config import VAULTS, get_vault
from src.indexer.embedder import get_collection, get_embedding_function, get_active_backend
from src.indexer.calibration import get_thresholds
from src.indexer.store import get_entity, get_observations, resolve_entity
from src.graph.traversal import get_neighbors, get_graph_boost_entity_ids

logger = logging.getLogger(__name__)

# Cache loaded calibrations
_calibration_cache: dict[str, dict] = {}
_SEARCH_INIT_GUARD_SECONDS = float(os.environ.get("MEMORY_SEARCH_INIT_TIMEOUT_SECONDS", "90"))

# Single-flight search backend init
_search_init_lock = threading.Lock()
_search_init_event: threading.Event | None = None
_search_init_thread: threading.Thread | None = None
_search_init_error: Exception | None = None
_search_init_started_at: float | None = None


def invalidate_calibration_cache(vault: str | None = None) -> None:
    if vault is None:
        _calibration_cache.clear()
    else:
        _calibration_cache.pop(vault, None)


def search_memory(query: str, vault: str = "", n_results: int = 10,
                   entity_type: str = "", include_neighbors: bool = True,
                   output_format: str = "text") -> str:
    """Hybrid vector + graph-boosted semantic memory search.

    Args:
        query: Natural language query describing what you're looking for.
        vault: Vault to search (empty = search all vaults).
        n_results: Number of results (default 10, max 30).
        entity_type: Optional entity type filter.
        include_neighbors: Include graph-connected entities (default True).
        output_format: "text" (default) or "json".

    Returns:
        Search results with entity details, observations, and confidence.
    """
    output_format = (output_format or "text").lower()
    if output_format not in {"text", "json"}:
        return "Error: output_format must be 'text' or 'json'."

    n_results = min(max(n_results, 1), 30)

    # Determine which vaults to search
    if vault:
        if vault not in VAULTS:
            return f"Error: Unknown vault '{vault}'. Use list_vaults() to see available vaults."
        vault_names = [vault]
    else:
        vault_names = list(VAULTS.keys())

    if not vault_names:
        return "No vaults configured. Use create_vault() to create one."

    # Get query embeddings
    try:
        query_embeddings = _get_query_embeddings_with_guard(query)
    except TimeoutError:
        return "Search backend is still initializing. Please retry in a few seconds."
    except Exception as e:
        return f"Search error: {e}"

    # Collect results from all target vaults
    all_items: list[dict] = []

    for vault_name in vault_names:
        vault_cfg = get_vault(vault_name)
        if vault_cfg is None:
            continue

        try:
            collection = get_collection(vault_cfg.collection_name)
        except Exception:
            continue

        # Build where clause
        where = None
        if entity_type:
            where = {"entity_type": entity_type}

        fetch_count = n_results * 3  # over-fetch for dedup
        try:
            results = collection.query(
                query_embeddings=query_embeddings,
                n_results=fetch_count,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.warning("Search error in vault %s: %s", vault_name, e)
            continue

        if not results["ids"] or not results["ids"][0]:
            continue

        ids = results["ids"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for i in range(len(ids)):
            meta = metadatas[i]
            all_items.append({
                "observation_id": ids[i],
                "entity_id": meta.get("entity_id", ""),
                "entity_name": meta.get("entity_name", ""),
                "entity_type": meta.get("entity_type", ""),
                "content": meta.get("content", ""),
                "source": meta.get("source", ""),
                "confidence": meta.get("confidence", 1.0),
                "vault": meta.get("vault", vault_name),
                "distance": distances[i],
                "graph_boosted": False,
            })

    if not all_items:
        return f"No results found for '{query}'."

    # Filter NO MATCH results
    all_items = [item for item in all_items
                 if _confidence_label(item["distance"], item["vault"]) != "NO MATCH"]
    if not all_items:
        return f"No confident results for '{query}' (all below relevance threshold)."

    # Deduplicate by entity — keep best observation per entity
    entity_best: dict[str, dict] = {}
    entity_observations: dict[str, list[dict]] = {}
    for item in all_items:
        eid = item["entity_id"]
        if eid not in entity_best or item["distance"] < entity_best[eid]["distance"]:
            entity_best[eid] = item
        if eid not in entity_observations:
            entity_observations[eid] = []
        entity_observations[eid].append(item)

    # Graph boost: expand via neighbors
    if include_neighbors and entity_best:
        result_entity_ids = set(entity_best.keys())
        boosted = get_graph_boost_entity_ids(result_entity_ids)

        for boosted_eid, boost_score in boosted.items():
            if boosted_eid in entity_best:
                continue
            ent = get_entity(boosted_eid)
            if ent is None:
                continue
            obs_list = get_observations(boosted_eid)
            if not obs_list:
                continue
            # Use the best observation's content as representative
            best_obs = obs_list[0]
            # Synthesize a result with boosted distance
            worst_distance = max(item["distance"] for item in all_items)
            synthetic_distance = worst_distance * (1.0 + (1.0 - boost_score))
            entity_best[boosted_eid] = {
                "observation_id": best_obs.id,
                "entity_id": ent.id,
                "entity_name": ent.name,
                "entity_type": ent.entity_type,
                "content": best_obs.content,
                "source": best_obs.source,
                "confidence": best_obs.confidence,
                "vault": ent.vault,
                "distance": synthetic_distance,
                "graph_boosted": True,
            }
            entity_observations[boosted_eid] = [
                {
                    "observation_id": o.id,
                    "content": o.content,
                    "source": o.source,
                }
                for o in obs_list
            ]

    # Sort by normalized score
    sorted_entities = sorted(
        entity_best.values(),
        key=lambda x: _normalized_score(x["distance"], x["vault"]),
        reverse=True,
    )

    # Trim to n_results
    sorted_entities = sorted_entities[:n_results]

    if output_format == "json":
        return _format_json(sorted_entities, entity_observations, query)
    return _format_text(sorted_entities, entity_observations, query)


def start_search_init() -> None:
    """Start search backend initialization eagerly."""
    if get_active_backend(role="search") == "not initialized":
        _ensure_search_backend_init_started()
        logger.info("Eager search init started at server startup")


def _get_query_embeddings_with_guard(query: str) -> list[list[float]]:
    """Get query embeddings with initialization guard."""
    if get_active_backend(role="search") != "not initialized":
        ef = get_embedding_function(role="search")
        return ef.embed_queries([query])

    event = _ensure_search_backend_init_started()
    if not event.wait(timeout=_SEARCH_INIT_GUARD_SECONDS):
        raise TimeoutError("Search backend initialization timed out.")
    if _search_init_error is not None:
        raise _search_init_error

    ef = get_embedding_function(role="search")
    return ef.embed_queries([query])


def _ensure_search_backend_init_started() -> threading.Event:
    """Start search backend init once per process."""
    global _search_init_event, _search_init_thread, _search_init_error, _search_init_started_at

    with _search_init_lock:
        if get_active_backend(role="search") != "not initialized":
            if _search_init_event is None:
                _search_init_event = threading.Event()
            _search_init_event.set()
            return _search_init_event

        if _search_init_event is not None and _search_init_thread is not None and _search_init_thread.is_alive():
            return _search_init_event

        _search_init_error = None
        _search_init_event = threading.Event()
        _search_init_started_at = time.monotonic()

        def _worker():
            global _search_init_error
            try:
                get_embedding_function(role="search")
                logger.info("Search backend initialization complete")
            except Exception as exc:
                _search_init_error = exc
                logger.exception("Search backend initialization failed")
            finally:
                if _search_init_event is not None:
                    _search_init_event.set()

        _search_init_thread = threading.Thread(
            target=_worker, daemon=True, name="memory-index-search-init"
        )
        _search_init_thread.start()
        return _search_init_event


def _get_thresholds_cached(vault: str) -> dict:
    if vault not in _calibration_cache:
        _calibration_cache[vault] = get_thresholds(vault)
    return _calibration_cache[vault]


def _confidence_label(distance: float, vault: str) -> str:
    thresholds = _get_thresholds_cached(vault)
    if distance < thresholds["HIGH"]:
        return "HIGH"
    elif distance < thresholds["MEDIUM"]:
        return "MEDIUM"
    elif distance < thresholds["LOW"]:
        return "LOW"
    else:
        return "NO MATCH"


def _normalized_score(distance: float, vault: str) -> float:
    """Piecewise normalized relevance score (0-100%)."""
    thresholds = _get_thresholds_cached(vault)
    high = thresholds.get("HIGH", 700)
    med = thresholds.get("MEDIUM", 800)
    low = thresholds.get("LOW", 900)
    floor = low * 1.2

    if distance <= 0:
        return 100.0
    elif distance <= high:
        return round(100.0 - (distance / high) * 40.0, 1)
    elif distance <= med:
        t = (distance - high) / (med - high) if med > high else 0
        return round(60.0 - t * 25.0, 1)
    elif distance <= low:
        t = (distance - med) / (low - med) if low > med else 0
        return round(35.0 - t * 20.0, 1)
    elif distance <= floor:
        t = (distance - low) / (floor - low) if floor > low else 0
        return round(max(0.0, 15.0 - t * 15.0), 1)
    else:
        return 0.0


def _format_text(results: list[dict], entity_obs: dict[str, list[dict]],
                 query: str) -> str:
    lines = []
    for i, item in enumerate(results):
        vault = item["vault"]
        confidence = _confidence_label(item["distance"], vault)
        score = _normalized_score(item["distance"], vault)
        boosted = " [graph-boosted]" if item.get("graph_boosted") else ""

        lines.append(f"--- Result {i + 1} (relevance: {score}%, confidence: {confidence}{boosted}) ---")
        lines.append(f"  Entity: {item['entity_name']} ({item['entity_type']})")
        lines.append(f"  Vault: {vault}")
        lines.append(f"  Entity ID: {item['entity_id']}")

        # Show matched observations
        obs_list = entity_obs.get(item["entity_id"], [])
        if obs_list:
            lines.append(f"  Observations ({len(obs_list)}):")
            for obs in obs_list[:5]:  # cap at 5 per entity
                src = f" [source: {obs.get('source', '')}]" if obs.get("source") else ""
                lines.append(f"    - {obs['content']}{src}")
        lines.append("")

    return "\n".join(lines)


def _format_json(results: list[dict], entity_obs: dict[str, list[dict]],
                 query: str) -> str:
    payload = {"query": query, "results": []}
    for i, item in enumerate(results):
        vault = item["vault"]
        obs_list = entity_obs.get(item["entity_id"], [])
        payload["results"].append({
            "result_num": i + 1,
            "entity_id": item["entity_id"],
            "entity_name": item["entity_name"],
            "entity_type": item["entity_type"],
            "vault": vault,
            "distance": round(float(item["distance"]), 4),
            "relevance_pct": _normalized_score(item["distance"], vault),
            "confidence": _confidence_label(item["distance"], vault),
            "graph_boosted": item.get("graph_boosted", False),
            "observations": [
                {"id": o.get("observation_id", ""), "content": o["content"],
                 "source": o.get("source", "")}
                for o in obs_list[:5]
            ],
        })
    return json.dumps(payload, indent=2)
