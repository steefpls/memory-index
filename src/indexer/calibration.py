"""Per-vault distance calibration for search confidence thresholds.

Samples real observations from the vault as "should match" probes, and uses
gibberish as "should not match" probes. Derives adaptive L2 distance thresholds
from the gap between the two distributions.
"""

import json
import logging
import random
from datetime import datetime, timezone

import numpy as np

from src.config import DATA_DIR
from src.indexer.embedder import get_embedding_function

logger = logging.getLogger(__name__)

# Fallback probes if the vault has too few observations to sample from
_FALLBACK_KNOWLEDGE_QUERIES = [
    "person responsible for project", "technology stack used",
    "decision made about architecture", "error encountered during deployment",
    "solution for performance issue", "configuration setting change",
    "dependency between components", "API design decision rationale",
    "database migration strategy", "testing approach for integration",
]

_NONSENSE_QUERIES = [
    "xkq7 zpmf bratl vvnx plrm",
    "aaaaa bbbbb ccccc ddddd eeeee",
    "12345 67890 !@#$% ^&*() +=<>",
    "the the the the the the the",
    "asdfghjkl qwertyuiop zxcvbnm",
]

_MIN_SAMPLES = 10  # minimum observations before sampling is useful

# Fallback thresholds if no calibration file exists.
# EmbeddingGemma vectors are L2-normalized, so Chroma's squared-L2 distance
# lives in [0, 4] (d = 2 - 2*cosine). These are rough priors; per-vault
# calibration replaces them on first run.
_DEFAULT_THRESHOLDS = {
    "HIGH": 0.6,
    "MEDIUM": 1.0,
    "LOW": 1.4,
}


def _sample_knowledge_queries(collection, n: int = 30) -> list[str]:
    """Sample real observation texts from the vault as calibration probes.

    Uses the actual stored documents so calibration reflects what's in the vault,
    not a hardcoded assumption about content domain.

    Sampling is uniform across the *whole* collection. `collection.peek(limit=n)`
    returns the n oldest rows in insertion order, which anchored every threshold
    to the vault's earliest facts and never moved as the vault grew. Instead we
    pull the full id list (ids only — cheap, no documents/embeddings) and
    `random.sample` from it, so calibration tracks the vault's current content.
    """
    total = collection.count()
    if total < _MIN_SAMPLES:
        return _FALLBACK_KNOWLEDGE_QUERIES

    docs: list[str] = []
    try:
        all_ids = (collection.get(include=[]) or {}).get("ids") or []
        if len(all_ids) >= _MIN_SAMPLES:
            sample_size = min(n, len(all_ids))
            chosen = random.sample(list(all_ids), sample_size)
            fetched = collection.get(ids=chosen, include=["documents"]) or {}
            docs = [d for d in (fetched.get("documents") or []) if d]
    except Exception as e:
        logger.warning("Random calibration sampling failed (%s); falling back to peek", e)
        docs = []

    if len(docs) < _MIN_SAMPLES:
        # Last resort before hardcoded probes: peek is biased toward the oldest
        # rows, but real vault content still beats generic queries.
        results = collection.peek(limit=min(n, total)) or {}
        docs = [d for d in (results.get("documents") or []) if d]

    if len(docs) < _MIN_SAMPLES:
        return _FALLBACK_KNOWLEDGE_QUERIES

    random.shuffle(docs)
    return docs[:n]


def calibrate_collection(collection, vault_name: str) -> dict:
    """Run calibration queries and save per-vault thresholds.

    Samples real observations as "should match" probes and queries them
    against the collection. With asymmetric query/document prefixes, a probe's
    nearest hit (usually itself) is NOT at distance ~0 — it measures the best
    distance a *perfect* query can achieve, which anchors the HIGH band.
    Anchoring bands on second-nearest distances (the old scheme) tied labels
    to vault density instead of match quality and made real paraphrase
    queries read as LOW. Gibberish probes establish the noise floor;
    MEDIUM sits midway between perfect-match and noise.
    """
    ef = get_embedding_function()

    knowledge_queries = _sample_knowledge_queries(collection)
    all_queries = knowledge_queries + _NONSENSE_QUERIES
    all_embeddings = ef.embed_queries(all_queries)

    knowledge_embeddings = all_embeddings[:len(knowledge_queries)]
    nonsense_embeddings = all_embeddings[len(knowledge_queries):]

    knowledge_result = collection.query(
        query_embeddings=knowledge_embeddings, n_results=2, include=["distances"],
    )
    self_distances = []      # nearest hit — the perfect-match anchor
    neighbor_distances = []  # second-nearest — vault density, diagnostics only
    for dists in knowledge_result["distances"]:
        if dists:
            self_distances.append(dists[0])
        if len(dists) >= 2:
            neighbor_distances.append(dists[1])

    nonsense_result = collection.query(
        query_embeddings=nonsense_embeddings, n_results=1, include=["distances"],
    )
    nonsense_distances = [
        dists[0] for dists in nonsense_result["distances"] if dists
    ]

    self_arr = np.array(self_distances)
    neighbor_arr = np.array(neighbor_distances) if neighbor_distances else self_arr
    nonsense_arr = np.array(nonsense_distances)

    self_p75 = float(np.percentile(self_arr, 75))
    neighbor_p50 = float(np.percentile(neighbor_arr, 50))
    nonsense_p25 = float(np.percentile(nonsense_arr, 25))

    # HIGH: as close as genuinely *related* content typically sits (a good
    # paraphrase query behaves like related content, so this band is reachable
    # — verbatim self-match distance is not). MEDIUM: midway from there to the
    # noise floor. LOW: the noise floor itself. The self-match p75 is a lower
    # guard so a near-duplicate-heavy vault can't drag HIGH below the best
    # distance a perfect query achieves.
    high = max(neighbor_p50, self_p75)
    low = nonsense_p25
    medium = high + 0.5 * (low - high)
    # Enforce ordering for degenerate distributions
    medium = max(medium, high * 1.15)
    low = max(low, medium * 1.15)

    def _dist_stats(arr):
        return {
            "min": round(float(arr.min()), 3),
            "p25": round(float(np.percentile(arr, 25)), 3),
            "p50": round(float(np.percentile(arr, 50)), 3),
            "p75": round(float(np.percentile(arr, 75)), 3),
            "max": round(float(arr.max()), 3),
        }

    calibration = {
        "vault_name": vault_name,
        "total_observations": collection.count(),
        "knowledge_probe_count": len(knowledge_queries),
        "knowledge_probe_source": "sampled" if collection.count() >= _MIN_SAMPLES else "fallback",
        "computed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        "self_distances": _dist_stats(self_arr),
        "neighbor_distances": _dist_stats(neighbor_arr),
        "nonsense_distances": _dist_stats(nonsense_arr),
        "thresholds": {
            "HIGH": round(high, 3),
            "MEDIUM": round(medium, 3),
            "LOW": round(low, 3),
        },
    }

    cal_path = DATA_DIR / f"{vault_name}_calibration.json"
    cal_path.parent.mkdir(parents=True, exist_ok=True)
    cal_path.write_text(json.dumps(calibration, indent=2), encoding="utf-8")

    logger.info("Calibration saved for %s: HIGH<%s MEDIUM<%s LOW<%s (source=%s, probes=%d)",
                vault_name, calibration["thresholds"]["HIGH"],
                calibration["thresholds"]["MEDIUM"], calibration["thresholds"]["LOW"],
                calibration["knowledge_probe_source"], len(knowledge_queries))

    return calibration


def load_calibration(vault_name: str) -> dict | None:
    """Load calibration JSON for a vault."""
    cal_path = DATA_DIR / f"{vault_name}_calibration.json"
    if not cal_path.exists():
        return None
    try:
        return json.loads(cal_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def get_thresholds(vault_name: str) -> dict:
    """Get thresholds for a vault, falling back to defaults."""
    cal = load_calibration(vault_name)
    if cal and "thresholds" in cal:
        return cal["thresholds"]
    return _DEFAULT_THRESHOLDS
