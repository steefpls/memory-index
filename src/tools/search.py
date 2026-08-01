"""Observation-level semantic memory search.

The retrieval unit is the *observation*, not the entity: every matching
observation is ranked flat by calibrated relevance, gated on the calibrated
noise threshold, and returned with its entity as context. Graph expansion
(strategy="associative") is opt-in and only ever admits neighbours that score
on their own merit against the query — no fabricated distances.
"""

import json
import logging
import os
import threading
from datetime import datetime, timezone

from src.config import VAULTS, get_vault
from src.indexer.embedder import get_collection, get_embedding_function, get_active_backend
from src.indexer.calibration import get_thresholds
from src.indexer.store import get_entity
from src.graph.traversal import spread_activation

logger = logging.getLogger(__name__)

# Cache loaded calibrations
_calibration_cache: dict[str, dict] = {}
_SEARCH_INIT_GUARD_SECONDS = float(os.environ.get("MEMORY_SEARCH_INIT_TIMEOUT_SECONDS", "90"))

# Serializes the one-time embedding backend load (single CPU backend).
_search_init_lock = threading.Lock()


def invalidate_calibration_cache(vault: str | None = None) -> None:
    if vault is None:
        _calibration_cache.clear()
    else:
        _calibration_cache.pop(vault, None)


DEFAULT_N_RESULTS = 5

# The tool never returns fewer than this many observations when the vault has
# anything to offer, even if nothing clears the relevance threshold. Honest
# LOW / NO MATCH labels are attached so the caller can judge for themselves.
MIN_RESULTS = 3


def search_memory(query: str, vault: str = "", n_results: int = DEFAULT_N_RESULTS,
                   entity_type: str = "",
                   since: str = "", before: str = "",
                   include_superseded: bool = False,
                   strategy: str = "semantic",
                   output_format: str = "text") -> str:
    """Semantic memory search over observations.

    Every matching observation is ranked flat by calibrated relevance (no
    dedupe-to-entity, no sampling of an entity's facts). Results above the
    calibrated noise threshold are returned, capped at n_results. If fewer than
    3 clear the threshold, the best 3 overall are returned anyway with their
    honest confidence labels.

    Args:
        query: Natural language query describing what you're looking for.
        vault: Vault to search (empty = search all vaults).
        n_results: Max observations to return (default 5, max 30).
        entity_type: Optional entity type filter.
        since: Only include observations created after this ISO date/datetime.
        before: Only include observations created before this ISO date/datetime.
        include_superseded: If True, include observations that have been replaced
                            by newer ones. Default False (only current facts).
        strategy: "semantic" (default — vector search only) or "associative"
                  (also probes graph neighbours of the hits, admitting only
                  those whose observations independently clear the threshold).
        output_format: "text" (default, entity-grouped) or "json" (flat list).

    Returns:
        Ranked observations with entity context, confidence band and score.
    """
    output_format = (output_format or "text").lower()
    if output_format not in {"text", "json"}:
        return "Error: output_format must be 'text' or 'json'."

    n_results = min(max(n_results, 1), 30)
    strategy = (strategy or "semantic").lower()
    if strategy not in {"associative", "semantic"}:
        strategy = "semantic"

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

    where_conditions = _build_where_conditions(entity_type)

    # Date bounds are applied in Python, NOT in the vector store's `where`
    # clause: `created_at` is stored as an ISO *string*, and Chroma's `$gte` /
    # `$lt` only accept numeric operands — pushing them down made every
    # date-filtered search raise inside the query and return nothing.
    if since and _parse_bound(since) is None:
        return f"Error: invalid 'since' date '{since}'. Use ISO format (YYYY-MM-DD or full ISO datetime)."
    if before and _parse_bound(before) is None:
        return f"Error: invalid 'before' date '{before}'. Use ISO format (YYYY-MM-DD or full ISO datetime)."
    date_bounds = (_parse_bound(since), _parse_bound(before))

    # Over-fetch so the threshold gate and the min-3 rule both have material
    # to work with after superseded observations are dropped.
    fetch_count = max(n_results * 3, MIN_RESULTS * 3)
    if date_bounds[0] is not None or date_bounds[1] is not None:
        # Post-filtering shrinks the top-k window, so widen it first.
        fetch_count = min(fetch_count * 4, 500)

    all_items: list[dict] = []
    for vault_name in vault_names:
        all_items.extend(_query_vault(
            vault_name, query_embeddings, fetch_count,
            where_conditions, include_superseded,
            date_bounds=date_bounds,
        ))

    if not all_items:
        return f"No results found for '{query}'."

    ranked = _rank(all_items)
    above = [item for item in ranked if item["confidence"] != "NO MATCH"]

    # Opt-in graph expansion. Neighbours are *really* scored against the query
    # embedding and admitted only if they clear the same threshold — a graph
    # hop can surface a fact, it can never manufacture relevance.
    if strategy == "associative":
        graph_items = _score_graph_candidates(
            seed_items=above or ranked,
            query_embeddings=query_embeddings,
            where_conditions=where_conditions,
            include_superseded=include_superseded,
            fetch_count=fetch_count,
            date_bounds=date_bounds,
        )
        if graph_items:
            seen = {item["observation_id"] for item in ranked}
            fresh = [g for g in graph_items if g["observation_id"] not in seen]
            if fresh:
                ranked = _rank(ranked + fresh)
                above = [item for item in ranked if item["confidence"] != "NO MATCH"]

    if len(above) >= MIN_RESULTS:
        selected = above[:n_results]
    else:
        # Min-3 rule: nothing (or almost nothing) cleared the bar — hand back
        # the best few anyway rather than an empty result, labels intact.
        selected = ranked[:MIN_RESULTS]

    if output_format == "json":
        return _format_json(selected, query, strategy=strategy,
                            above_threshold_count=len(above))
    return _format_text(selected, query, above_threshold_count=len(above))


def _build_where_conditions(entity_type: str) -> list[dict]:
    """Metadata filter conditions shared by the semantic and graph passes.

    ONLY equality/`$in` predicates belong here. Chroma's range operators
    (`$gte`, `$lt`) require int/float operands, and `created_at` is stored as
    an ISO string — pushing a date bound down raises inside `collection.query`
    and silently yields zero results. Date windows are applied by
    `_in_date_range` after the query instead.
    """
    conditions: list[dict] = []
    if entity_type:
        conditions.append({"entity_type": entity_type})
    return conditions


def _parse_bound(value: str) -> datetime | None:
    """Parse an ISO date/datetime bound into a UTC-aware datetime, or None."""
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except (ValueError, TypeError):
        try:
            dt = datetime.strptime(value, "%Y-%m-%d")
        except (ValueError, TypeError):
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _in_date_range(meta: dict, date_bounds: tuple) -> bool:
    """Apply the since/before window to an observation's `created_at`.

    An observation whose `created_at` cannot be parsed is excluded whenever a
    bound is active — it cannot be shown to satisfy the window.
    """
    since_dt, before_dt = date_bounds
    if since_dt is None and before_dt is None:
        return True
    created = _parse_bound(meta.get("created_at") or "")
    if created is None:
        return False
    if since_dt is not None and created < since_dt:
        return False
    if before_dt is not None and created >= before_dt:
        return False
    return True


def _combine_where(conditions: list[dict]) -> dict | None:
    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def _query_vault(vault_name: str, query_embeddings, fetch_count: int,
                 where_conditions: list[dict], include_superseded: bool,
                 entity_ids: list[str] | None = None,
                 graph_boosted: bool = False,
                 date_bounds: tuple = (None, None)) -> list[dict]:
    """Run one Chroma query and return flat observation items.

    entity_ids restricts the query to a candidate set (used by the associative
    pass to score graph neighbours with the real query embedding).
    """
    vault_cfg = get_vault(vault_name)
    if vault_cfg is None:
        return []

    try:
        collection = get_collection(vault_cfg.collection_name)
    except Exception:
        return []

    conditions = list(where_conditions)
    if entity_ids:
        conditions.append({"entity_id": {"$in": list(entity_ids)}})

    try:
        results = collection.query(
            query_embeddings=query_embeddings,
            n_results=fetch_count,
            where=_combine_where(conditions),
            include=["documents", "metadatas", "distances"],
        )
    except Exception as e:
        logger.warning("Search error in vault %s: %s", vault_name, e)
        return []

    ids_batches = results.get("ids") or []
    if not ids_batches or not ids_batches[0]:
        return []

    ids = ids_batches[0]
    metadatas = (results.get("metadatas") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]

    items: list[dict] = []
    for i in range(len(ids)):
        meta = metadatas[i] or {}
        is_superseded = bool(meta.get("superseded_by"))
        # Skip superseded observations unless explicitly requested
        if is_superseded and not include_superseded:
            continue
        # since/before window — applied here, not in the `where` clause.
        if not _in_date_range(meta, date_bounds):
            continue
        items.append({
            "observation_id": ids[i],
            "entity_id": meta.get("entity_id", ""),
            "entity_name": meta.get("entity_name", ""),
            "entity_type": meta.get("entity_type", ""),
            "content": meta.get("content", ""),
            "source": meta.get("source", ""),
            "vault": meta.get("vault", vault_name),
            "distance": distances[i],
            "graph_boosted": graph_boosted,
            "superseded": is_superseded,
        })
    return items


def _rank(items: list[dict]) -> list[dict]:
    """Annotate every observation with its calibrated score/band and sort.

    Score is used as the sort key rather than raw distance because it is
    calibrated per vault — the only way cross-vault results compare fairly.
    Within a single vault it is a strictly decreasing function of distance,
    so this is exactly "best distance first".
    """
    for item in items:
        item["score"] = _normalized_score(item["distance"], item["vault"])
        item["confidence"] = _confidence_label(item["distance"], item["vault"])
    return sorted(items, key=lambda x: (-x["score"], x["distance"]))


def _score_graph_candidates(seed_items: list[dict], query_embeddings,
                            where_conditions: list[dict],
                            include_superseded: bool,
                            fetch_count: int,
                            date_bounds: tuple = (None, None)) -> list[dict]:
    """Score graph neighbours of the semantic hits against the real query.

    Spreading activation only *nominates* candidate entities. Their
    observations are then queried with the actual query embedding and admitted
    only if they clear the same noise threshold as a direct hit. Nothing here
    invents a distance, so a well-connected entity cannot outrank a genuine
    semantic match. Returns [] when no neighbour earns its place.
    """
    seed_ids = {item["entity_id"] for item in seed_items if item.get("entity_id")}
    if not seed_ids:
        return []

    try:
        activated = spread_activation(
            seed_ids=seed_ids, decay=0.7, max_hops=3, top_k=10,
        )
    except Exception as e:
        logger.warning("Spreading activation failed: %s", e)
        return []

    candidates_by_vault: dict[str, list[str]] = {}
    for candidate_id in activated:
        if candidate_id in seed_ids:
            continue
        ent = get_entity(candidate_id)
        if ent is None:
            continue
        candidates_by_vault.setdefault(ent.vault, []).append(candidate_id)

    if not candidates_by_vault:
        return []

    scored: list[dict] = []
    for vault_name, entity_ids in candidates_by_vault.items():
        scored.extend(_query_vault(
            vault_name, query_embeddings, fetch_count,
            where_conditions, include_superseded,
            entity_ids=entity_ids, graph_boosted=True,
            date_bounds=date_bounds,
        ))

    return [item for item in scored
            if _confidence_label(item["distance"], item["vault"]) != "NO MATCH"]


def start_search_init() -> None:
    """Warm the embedding backend at server startup, off the request path.

    Model load takes seconds; doing it here means the first search does not pay
    for it. Failures are logged and swallowed — the next `_ensure_backend` call
    retries and surfaces the error to the caller that actually needs it.
    """
    if get_active_backend() != "not initialized":
        return

    def _warm():
        try:
            _ensure_backend()
            logger.info("Eager search backend init complete")
        except Exception:
            logger.exception("Eager search backend init failed")

    threading.Thread(target=_warm, daemon=True,
                     name="memory-index-search-init").start()
    logger.info("Eager search init started at server startup")


def _ensure_backend():
    """Return the embedding singleton, initializing it if needed.

    Single-construction is guaranteed by `_embedder_lock` inside
    `embedder.get_embedding_function()` — that is the only lock every caller
    (store writes, calibration, the auto-librarian thread) shares. This lock
    exists purely so a *search* caller can time out with a useful message
    rather than sitting behind a wedged model load forever.
    """
    if get_active_backend() != "not initialized":
        return get_embedding_function()

    if not _search_init_lock.acquire(timeout=_SEARCH_INIT_GUARD_SECONDS):
        raise TimeoutError(
            f"Search backend initialization timed out after "
            f"{_SEARCH_INIT_GUARD_SECONDS:.0f}s."
        )
    try:
        return get_embedding_function()
    finally:
        _search_init_lock.release()


def _get_query_embeddings_with_guard(query: str) -> list[list[float]]:
    """Embed a query, initializing the backend on first use."""
    return _ensure_backend().embed_queries([query])


def _rrf_merge(vector_ranked: list[dict], graph_ranked: list[dict],
               k: int = 60) -> dict[str, float]:
    """Reciprocal Rank Fusion — merge vector and graph result rankings.

    NOTE: no longer used by search_memory. RRF is rank-only, so it has no
    notion of match quality: a popularity-ranked graph neighbour could outrank
    a genuine semantic hit. The associative strategy now scores graph
    candidates against the real query embedding instead. Kept as a utility.

    Combines rankings from two sources using RRF scoring:
      score(entity) = sum(1 / (k + rank_i)) across all lists.

    Args:
        vector_ranked: Entities ranked by vector distance (best first).
        graph_ranked: Entities ranked by activation energy (best first).
        k: Smoothing constant (default 60, standard RRF value).

    Returns:
        Dict mapping entity_id -> RRF score (higher is better).
    """
    scores: dict[str, float] = {}

    for rank, item in enumerate(vector_ranked, 1):
        eid = item["entity_id"]
        scores[eid] = scores.get(eid, 0.0) + 1.0 / (k + rank)

    for rank, item in enumerate(graph_ranked, 1):
        eid = item["entity_id"]
        scores[eid] = scores.get(eid, 0.0) + 1.0 / (k + rank)

    return scores


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
    """Piecewise normalized relevance score (0-100%).

    Bands are anchored to calibrated thresholds so that real matches
    land in an intuitive range:
      distance 0        → 100%
      distance = HIGH   → 85%   (knowledge p25 — top-quartile match)
      distance = MEDIUM → 55%   (knowledge p75 — typical match)
      distance = LOW    → 15%   (nonsense p25  — noise floor)
      distance > floor  →  0%
    """
    thresholds = _get_thresholds_cached(vault)
    high = thresholds.get("HIGH", 0.6)
    med = thresholds.get("MEDIUM", 1.0)
    low = thresholds.get("LOW", 1.4)
    floor = low * 1.2

    if distance <= 0:
        return 100.0
    elif distance <= high:
        return round(100.0 - (distance / high) * 15.0, 1)
    elif distance <= med:
        t = (distance - high) / (med - high) if med > high else 0
        return round(85.0 - t * 30.0, 1)
    elif distance <= low:
        t = (distance - med) / (low - med) if low > med else 0
        return round(55.0 - t * 40.0, 1)
    elif distance <= floor:
        t = (distance - low) / (floor - low) if floor > low else 0
        return round(max(0.0, 15.0 - t * 15.0), 1)
    else:
        return 0.0


_CONFIDENCE_SHORT = {"HIGH": "HIGH", "MEDIUM": "MED", "LOW": "LOW"}


def _format_text(results: list[dict], query: str,
                 above_threshold_count: int | None = None) -> str:
    """Render the flat ranking, grouping consecutive same-entity hits.

    Ranking stays per-observation; the entity header is purely a token-saving
    device so a run of facts about one entity doesn't repeat its context line.
    """
    lines: list[str] = []

    if above_threshold_count is not None and above_threshold_count < len(results):
        lines.append(
            f"note: {above_threshold_count} result(s) cleared the relevance "
            f"threshold — showing the best {len(results)} overall."
        )

    last_key = None
    for i, item in enumerate(results):
        vault = item["vault"]
        key = (item.get("entity_id", ""), vault)
        if key != last_key:
            lines.append(
                f"{item.get('entity_name', '')} ({item.get('entity_type', '')}) · "
                f"{vault} · id={item.get('entity_id', '')}"
            )
            last_key = key

        confidence = _confidence_label(item["distance"], vault)
        conf_short = _CONFIDENCE_SHORT.get(confidence, confidence)
        score = _normalized_score(item["distance"], vault)
        boosted = " +graph" if item.get("graph_boosted") else ""
        src = f" [src: {item['source']}]" if item.get("source") else ""
        old = " [superseded]" if item.get("superseded") else ""

        lines.append(
            f"  [{i + 1}] {score}% {conf_short}{boosted} · "
            f"{item.get('content', '')}{src}{old}"
        )

    return "\n".join(lines)


def _format_json(results: list[dict], query: str, strategy: str = "semantic",
                 above_threshold_count: int | None = None) -> str:
    """Flat ranked list of observation objects — no entity nesting."""
    payload = {
        "query": query,
        "strategy": strategy,
        "returned": len(results),
        "above_threshold": above_threshold_count,
        "results": [],
    }
    for i, item in enumerate(results):
        vault = item["vault"]
        confidence = _confidence_label(item["distance"], vault)
        payload["results"].append({
            "rank": i + 1,
            "observation_id": item.get("observation_id", ""),
            "content": item.get("content", ""),
            "source": item.get("source", ""),
            "entity_id": item.get("entity_id", ""),
            "entity_name": item.get("entity_name", ""),
            "entity_type": item.get("entity_type", ""),
            "vault": vault,
            "distance": round(float(item["distance"]), 4),
            "relevance_pct": _normalized_score(item["distance"], vault),
            "confidence": confidence,
            "above_threshold": confidence != "NO MATCH",
            "graph_boosted": bool(item.get("graph_boosted")),
            "superseded": bool(item.get("superseded")),
        })
    return json.dumps(payload, indent=2)
