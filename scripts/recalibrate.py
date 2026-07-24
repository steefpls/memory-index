"""Recalibrate distance thresholds for all vaults without re-embedding.

Run after calibration-scheme changes, or whenever confidence labels feel off.
Safe to run while the service is stopped; the service also recalibrates
automatically every N observations.

Usage: python scripts/recalibrate.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import list_vaults
from src.indexer.embedder import get_collection
from src.indexer.calibration import calibrate_collection


def main() -> int:
    for v in list_vaults():
        collection = get_collection(v.collection_name)
        if collection.count() == 0:
            print(f"[{v.name}] empty — skipped")
            continue
        cal = calibrate_collection(collection, v.name)
        th = cal["thresholds"]
        print(f"[{v.name}] recalibrated ({cal['total_observations']} obs): "
              f"HIGH<{th['HIGH']} MEDIUM<{th['MEDIUM']} LOW<{th['LOW']}")
        print(f"    self-match p50={cal['self_distances']['p50']}, "
              f"noise floor p25={cal['nonsense_distances']['p25']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
