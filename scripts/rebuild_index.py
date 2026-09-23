"""Rebuild every vault's search index (data/chroma) from data/memory.db.

For a restore. The fleet backup keeps memory.db -- every vault's entities,
observations and relations -- but not the vectors, which the embedder can
make again. After memory.db is put back the index is empty and every search
finds nothing; this fills it: each entity's active observations are
re-embedded (the same path as the reembed_entity tool), then each vault is
recalibrated.

reembed_all.py is the other repair and does not help here: it re-embeds
what is already IN the index, for a model swap.

Run with the service stopped, from the checkout, on the venv the service
starts with:
    .venv\\Scripts\\python.exe scripts\\rebuild_index.py [--check]

--check only compares each vault's index to memory.db and exits 1 when an
index holds fewer vectors than the vault has observations.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import list_vaults  # noqa: E402
from src.indexer.calibration import calibrate_collection  # noqa: E402
from src.indexer.embedder import get_collection  # noqa: E402
from src.indexer.store import get_observations, list_entities, reembed_entity  # noqa: E402


def vault_counts(vault: str) -> tuple[list, int]:
    entities, total = list_entities(vault=vault, limit=10**9)
    observations = sum(len(get_observations(e.id)) for e in entities)
    return entities, observations


def main() -> int:
    check = "--check" in sys.argv
    vaults = list_vaults()
    if not vaults:
        print("No vaults in memory.db -- nothing to rebuild.")
        return 0

    short = False
    t0 = time.time()
    for v in vaults:
        collection = get_collection(v.collection_name)
        entities, observations = vault_counts(v.name)
        have = collection.count()
        print(f"[{v.name}] {len(entities)} entities, {observations} active observations; "
              f"the index holds {have} vectors")
        if check:
            short = short or have < observations
            continue
        if not observations:
            continue
        done = failed = 0
        for i, e in enumerate(entities, 1):
            _, (ok, bad) = reembed_entity(e.id)
            done += ok
            failed += bad
            print(f"    {i}/{len(entities)} entities, {done} observations embedded", end="\r", flush=True)
        print()
        cal = calibrate_collection(collection, v.name)
        th = cal["thresholds"]
        print(f"[{v.name}] {done} embedded, {failed} failed; the index now holds {collection.count()} vectors; "
              f"recalibrated HIGH<{th['HIGH']} MEDIUM<{th['MEDIUM']} LOW<{th['LOW']}")
        short = short or failed > 0

    if check:
        print("Every index is complete." if not short else "An index is short of its vault.")
    else:
        print(f"Done in {time.time() - t0:.1f}s" + (" -- some observations failed, run it again" if short else ""))
    return 1 if short else 0


if __name__ == "__main__":
    sys.exit(main())
