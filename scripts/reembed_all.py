"""Re-embed every vault's ChromaDB collection with the current embedding model.

Run once after an embedding-model swap (old and new vectors cannot coexist in
one collection). Iterates the collections directly rather than going through
entities so superseded observations get re-embedded too, then recalibrates
each vault's distance thresholds for the new vector space.

Usage: python scripts/reembed_all.py [--dry-run]
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import list_vaults, DATA_DIR
from src.indexer.embedder import get_embedding_function, get_collection
from src.indexer.calibration import calibrate_collection

_BATCH = 32


def reembed_collection(collection, ef) -> int:
    total = collection.count()
    if total == 0:
        return 0

    done = 0
    offset = 0
    while offset < total:
        page = collection.get(
            limit=_BATCH, offset=offset,
            include=["documents", "metadatas"],
        )
        ids = page["ids"]
        if not ids:
            break
        docs = page["documents"]
        metas = page["metadatas"]

        # Stored documents are the raw embedding text; ef() applies the
        # model's document prefix itself.
        embeddings = ef(docs)
        collection.upsert(ids=ids, embeddings=embeddings,
                          documents=docs, metadatas=metas)

        done += len(ids)
        offset += len(ids)
        print(f"    {done}/{total}", end="\r", flush=True)

    print()
    return done


def main() -> int:
    dry_run = "--dry-run" in sys.argv

    vaults = list_vaults()
    if not vaults:
        print("No vaults configured — nothing to do.")
        return 0

    print(f"Vaults: {[v.name for v in vaults]}")
    if dry_run:
        for v in vaults:
            print(f"  {v.name}: {get_collection(v.collection_name).count()} vectors")
        return 0

    ef = get_embedding_function()
    print(f"Embedder backend: {ef.backend}\n")

    t0 = time.time()
    for v in vaults:
        collection = get_collection(v.collection_name)
        print(f"[{v.name}] re-embedding {collection.count()} observations...")
        n = reembed_collection(collection, ef)
        if n > 0:
            cal = calibrate_collection(collection, v.name)
            th = cal["thresholds"]
            print(f"[{v.name}] done ({n} re-embedded), recalibrated: "
                  f"HIGH<{th['HIGH']} MEDIUM<{th['MEDIUM']} LOW<{th['LOW']}")
        else:
            # Stale calibration from the old vector space must not linger
            cal_path = DATA_DIR / f"{v.name}_calibration.json"
            if cal_path.exists():
                cal_path.unlink()
            print(f"[{v.name}] empty — skipped (stale calibration removed)")

    print(f"\nAll vaults re-embedded in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
