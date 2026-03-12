"""Per-vault distance calibration for search confidence thresholds.

After adding enough observations, runs generic knowledge + nonsense queries
against the collection to derive adaptive L2 distance thresholds.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.config import DATA_DIR
from src.indexer.embedder import get_embedding_function

logger = logging.getLogger(__name__)

# Memory-domain calibration queries (knowledge/facts instead of code)
_KNOWLEDGE_QUERIES = [
    "person responsible for project", "technology stack used",
    "decision made about architecture", "error encountered during deployment",
    "solution for performance issue", "configuration setting change",
    "dependency between components", "meeting notes from review",
    "API design decision rationale", "database migration strategy",
    "security vulnerability fix", "testing approach for integration",
    "deployment pipeline setup", "monitoring and alerting rules",
    "user feedback on feature", "team member expertise area",
    "project deadline milestone", "infrastructure cost optimization",
    "code review feedback pattern", "documentation update needed",
    "bug report root cause", "feature request priority",
    "environment setup instructions", "third party library evaluation",
    "performance benchmark results", "data model schema change",
    "authentication flow design", "error handling strategy",
    "caching layer implementation", "logging and observability",
]

_NONSENSE_QUERIES = [
    "chocolate cake recipe frosting", "weather forecast tomorrow rain",
    "guitar chord progression blues", "gardening tips growing tomatoes",
    "movie review rating stars",
]

# Fallback thresholds if no calibration file exists
_DEFAULT_THRESHOLDS = {
    "HIGH": 650,
    "MEDIUM": 775,
    "LOW": 875,
}


def calibrate_collection(collection, vault_name: str) -> dict:
    """Run calibration queries and save per-vault thresholds."""
    ef = get_embedding_function(role="index")

    all_queries = _KNOWLEDGE_QUERIES + _NONSENSE_QUERIES
    all_embeddings = ef.embed_queries(all_queries)

    knowledge_embeddings = all_embeddings[:len(_KNOWLEDGE_QUERIES)]
    nonsense_embeddings = all_embeddings[len(_KNOWLEDGE_QUERIES):]

    knowledge_result = collection.query(
        query_embeddings=knowledge_embeddings, n_results=1, include=["distances"],
    )
    knowledge_distances = [
        dists[0] for dists in knowledge_result["distances"] if dists
    ]

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

    logger.info("Calibration saved for %s: HIGH<%s MEDIUM<%s LOW<%s",
                vault_name, calibration["thresholds"]["HIGH"],
                calibration["thresholds"]["MEDIUM"], calibration["thresholds"]["LOW"])

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
