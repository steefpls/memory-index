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

# Fallback thresholds if no calibration file exists
_DEFAULT_THRESHOLDS = {
    "HIGH": 650,
    "MEDIUM": 775,
    "LOW": 875,
}


def _sample_knowledge_queries(collection, n: int = 30) -> list[str]:
    """Sample real observation texts from the vault as calibration probes.

    Uses the actual stored documents so calibration reflects what's in the vault,
    not a hardcoded assumption about content domain.
    """
    total = collection.count()
    if total < _MIN_SAMPLES:
        return _FALLBACK_KNOWLEDGE_QUERIES

    # Peek returns up to `limit` documents from the collection
    sample_size = min(n, total)
    results = collection.peek(limit=sample_size)

    docs = results.get("documents") or []
    if len(docs) < _MIN_SAMPLES:
        return _FALLBACK_KNOWLEDGE_QUERIES

    # Shuffle so we don't always get the same subset if vault grows
    random.shuffle(docs)
    return docs[:n]


def calibrate_collection(collection, vault_name: str) -> dict:
    """Run calibration queries and save per-vault thresholds.

    Samples real observations as "should match" probes and queries them
    against the collection (they'll match themselves or similar entries).
    Gibberish probes establish the noise floor.
    """
    ef = get_embedding_function(role="index")

    knowledge_queries = _sample_knowledge_queries(collection)
    all_queries = knowledge_queries + _NONSENSE_QUERIES
    all_embeddings = ef.embed_queries(all_queries)

    knowledge_embeddings = all_embeddings[:len(knowledge_queries)]
    nonsense_embeddings = all_embeddings[len(knowledge_queries):]

    # n_results=2 for knowledge probes: the closest match will be the
    # observation itself (distance ~0), so we want the second-nearest
    # to measure how close *other* content is.
    knowledge_result = collection.query(
        query_embeddings=knowledge_embeddings, n_results=2, include=["distances"],
    )
    knowledge_distances = []
    for dists in knowledge_result["distances"]:
        if len(dists) >= 2:
            knowledge_distances.append(dists[1])  # second-nearest
        elif dists:
            knowledge_distances.append(dists[0])

    nonsense_result = collection.query(
        query_embeddings=nonsense_embeddings, n_results=1, include=["distances"],
    )
    nonsense_distances = [
        dists[0] for dists in nonsense_result["distances"] if dists
    ]

    knowledge_arr = np.array(knowledge_distances)
    nonsense_arr = np.array(nonsense_distances)

    knowledge_p25 = float(np.percentile(knowledge_arr, 25))
    knowledge_p75 = float(np.percentile(knowledge_arr, 75))
    nonsense_p25 = float(np.percentile(nonsense_arr, 25))

    calibration = {
        "vault_name": vault_name,
        "total_observations": collection.count(),
        "knowledge_probe_count": len(knowledge_queries),
        "knowledge_probe_source": "sampled" if collection.count() >= _MIN_SAMPLES else "fallback",
        "computed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        "knowledge_distances": {
            "min": float(knowledge_arr.min()),
            "p25": round(float(np.percentile(knowledge_arr, 25)), 1),
            "p50": round(float(np.percentile(knowledge_arr, 50)), 1),
            "p75": round(float(np.percentile(knowledge_arr, 75)), 1),
            "max": float(knowledge_arr.max()),
        },
        "nonsense_distances": {
            "min": float(nonsense_arr.min()),
            "p25": round(float(np.percentile(nonsense_arr, 25)), 1),
            "p50": round(float(np.percentile(nonsense_arr, 50)), 1),
            "p75": round(float(np.percentile(nonsense_arr, 75)), 1),
            "max": float(nonsense_arr.max()),
        },
        "thresholds": {
            "HIGH": round(knowledge_p25, 1),
            "MEDIUM": round(knowledge_p75, 1),
            "LOW": round(nonsense_p25, 1),
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
