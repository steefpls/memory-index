"""Maintenance tools — vacuum stale rows, repair dangling references."""

import logging

from src.config import VAULTS
from src.indexer import store as store_mod
from src.graph import manager as graph_mod

logger = logging.getLogger(__name__)


def tool_vacuum_store(dry_run: bool = False) -> str:
    """Hard-remove stale rows that accumulated despite soft-delete semantics.

    Targets:
      - Entities whose vault no longer exists in VAULTS
      - Soft-deleted entities (already invisible to all queries)
      - Observations whose entity_id is no longer present after the above
      - Relations whose endpoints reference an entity that no longer exists

    The Chroma vectors for affected observations are best-effort removed.
    Run with dry_run=True first to preview what would be removed.

    Args:
        dry_run: If True, report what would be removed without modifying state.
    """
    store_mod._load_store()
    graph_mod._get_graph()

    valid_vaults = set(VAULTS.keys())

    # ---- Entities to hard-remove ----
    ent_to_remove: list[str] = []
    ent_orphan_vault = 0
    ent_soft_deleted = 0
    chroma_targets: dict[str, list[str]] = {}  # collection_name -> obs ids

    for ent in list(store_mod._entities.values()):
        reason = None
        if ent.vault not in valid_vaults:
            reason = "orphan_vault"
            ent_orphan_vault += 1
        elif ent.deleted:
            reason = "soft_deleted"
            ent_soft_deleted += 1
        if reason is None:
            continue
        ent_to_remove.append(ent.id)

        # Collect Chroma cleanup targets even for vaults that still exist
        if ent.vault in valid_vaults:
            collection_name = VAULTS[ent.vault].collection_name
            obs_ids = [
                o.id for o in store_mod._observations.values()
                if o.entity_id == ent.id
            ]
            if obs_ids:
                chroma_targets.setdefault(collection_name, []).extend(obs_ids)

    ent_to_remove_set = set(ent_to_remove)

    # ---- Observations whose entity is gone (now or already) ----
    surviving_ent_ids = {
        eid for eid in store_mod._entities.keys() if eid not in ent_to_remove_set
    }
    obs_to_remove = [
        o.id for o in store_mod._observations.values()
        if o.entity_id not in surviving_ent_ids
    ]

    # ---- Dangling relations ----
    rel_to_remove = []
    for rel in graph_mod.get_all_relations():
        if rel.from_entity not in surviving_ent_ids or rel.to_entity not in surviving_ent_ids:
            rel_to_remove.append(rel.id)

    summary_lines = [
        "Vacuum " + ("plan" if dry_run else "result") + ":",
        f"  Entities to remove: {len(ent_to_remove)} "
        f"({ent_orphan_vault} orphan vault, {ent_soft_deleted} soft-deleted)",
        f"  Observations to remove: {len(obs_to_remove)}",
        f"  Relations to remove: {len(rel_to_remove)}",
    ]

    if dry_run:
        return "\n".join(summary_lines + ["  (dry run — nothing modified)"])

    # ---- Execute ----
    # Best-effort Chroma cleanup
    if chroma_targets:
        try:
            from src.indexer.embedder import get_chroma_client
            client = get_chroma_client()
            for collection_name, ids in chroma_targets.items():
                try:
                    coll = client.get_collection(collection_name)
                    coll.delete(ids=ids)
                except Exception as e:
                    logger.warning("Chroma cleanup failed for %s: %s", collection_name, e)
        except Exception as e:
            logger.warning("Chroma client unavailable for vacuum: %s", e)

    for oid in obs_to_remove:
        store_mod._observations.pop(oid, None)

    for eid in ent_to_remove:
        store_mod._entities.pop(eid, None)

    for rid in rel_to_remove:
        graph_mod.remove_relation(rid)

    store_mod._save_store()
    # remove_relation already saves the graph, but call once more in case
    # rel_to_remove was empty (no save) but we still want to be sure.
    graph_mod._save_graph()

    logger.info(
        "Vacuum: removed %d entities, %d observations, %d relations",
        len(ent_to_remove), len(obs_to_remove), len(rel_to_remove),
    )

    return "\n".join(summary_lines)
