"""Vault import/export — bundle a vault to a zip file and restore it elsewhere.

The export format is a zip containing JSON files. Vector embeddings are NOT
included; they are regenerated on import via the embedder. This keeps the
archive portable across embedding model changes and keeps file size small.

Format (version 2 — v1 archives are still importable; v2 adds observation
superseded_at):
    manifest.json     — format version, source vault, timestamp, counts
    entities.json     — list of entity dicts for the source vault
    observations.json — list of observation dicts whose entity is in the vault
    relations.json    — list of relation dicts where both endpoints are in the vault

Import is always ADDITIVE: existing data in the target vault is preserved.
Entities are merged by name; observations are deduped by exact content per
entity; relations are deduped by (from, to, type).

Superseded observations survive the round-trip: they are imported like any
other row and then re-flagged as superseded (with their superseded_by pointers
remapped to the new local IDs), so history is not silently dropped and old
facts never resurface as current ones.

The supersede pass is bounded by the additive contract: it only ever retires
rows this import created. A row that deduped onto data the target vault
already had — or onto a row an active archive entry also claims — stays
active, because import must never hide a fact the user already stored.
"""

import json
import logging
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from src.config import (
    DATA_DIR, get_vault, create_vault as config_create_vault,
)
from src.indexer import store as store_mod
from src.indexer.store import (
    create_entity, get_entity_by_name, add_observation, get_observations,
    mark_superseded,
)
from src.graph import manager as graph_mod
from src.graph.manager import (
    get_all_relations, add_relation, get_relations_for_entity,
)
from src.models.relation import Relation

logger = logging.getLogger(__name__)

FORMAT_VERSION = 2  # v2 adds observation superseded_at
SUPPORTED_FORMAT_VERSIONS = {1, 2}


# ---------- Export ----------

def _collect_vault_data(vault: str) -> dict:
    """Gather entities, observations, and relations belonging to a vault.

    Includes soft-deleted entities/observations so an export is a complete
    snapshot. Relations are included only if BOTH endpoints are in the vault.
    """
    store_mod._load_store()
    graph_mod._get_graph()  # ensure relations are loaded

    # Snapshot under STORE_LOCK: a concurrent write (or the background
    # auto-librarian) mutating the live dicts mid-scan raised
    # "dictionary changed size during iteration".
    entities_snapshot, observations_snapshot = store_mod.snapshot_store()

    entities = [
        e.to_dict() for e in entities_snapshot.values()
        if e.vault == vault
    ]
    entity_ids = {e["id"] for e in entities}

    observations = [
        o.to_dict() for o in observations_snapshot.values()
        if o.entity_id in entity_ids
    ]

    relations = [
        r.to_dict() for r in get_all_relations()
        if r.from_entity in entity_ids and r.to_entity in entity_ids
    ]

    return {
        "entities": entities,
        "observations": observations,
        "relations": relations,
    }


def _resolve_export_path(vault: str, output_path: str) -> Path:
    """Decide where to write the export zip.

    - empty:      data/exports/<vault>_<timestamp>.zip
    - directory:  <output_path>/<vault>_<timestamp>.zip
    - file path:  <output_path> (parent dirs created)
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    default_name = f"{vault}_{timestamp}.zip"

    if not output_path:
        target = DATA_DIR / "exports" / default_name
    else:
        target = Path(output_path).expanduser()
        # Treat as directory if it lacks a .zip suffix, already is a directory,
        # or ends with a path separator.
        looks_like_dir = (
            target.is_dir()
            or output_path.endswith(("/", "\\"))
            or target.suffix.lower() != ".zip"
        )
        if looks_like_dir:
            target = target / default_name

    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def tool_export_vault(vault: str, output_path: str = "") -> str:
    """Export a vault to a zip file containing JSON data.

    Args:
        vault: Vault name to export.
        output_path: Optional output path. May be a directory (filename
                     auto-generated), an explicit .zip path, or empty (defaults
                     to data/exports/).

    Returns:
        Confirmation with target path and counts, or error.
    """
    if not vault or not vault.strip():
        return "Error: vault is required."

    vault = vault.strip()
    if get_vault(vault) is None:
        return f"Vault '{vault}' not found."

    data = _collect_vault_data(vault)

    manifest = {
        "format_version": FORMAT_VERSION,
        "source_vault": vault,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "counts": {
            "entities": len(data["entities"]),
            "observations": len(data["observations"]),
            "relations": len(data["relations"]),
        },
    }

    target = _resolve_export_path(vault, output_path)

    try:
        with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("manifest.json", json.dumps(manifest, indent=2))
            zf.writestr("entities.json", json.dumps(data["entities"], indent=2))
            zf.writestr("observations.json", json.dumps(data["observations"], indent=2))
            zf.writestr("relations.json", json.dumps(data["relations"], indent=2))
    except OSError as e:
        return f"Error writing export: {e}"

    size_kb = target.stat().st_size / 1024
    logger.info("Exported vault '%s' to %s (%.1f KB)", vault, target, size_kb)
    return (
        f"Exported vault '{vault}' to:\n"
        f"  {target}\n"
        f"  {manifest['counts']['entities']} entities, "
        f"{manifest['counts']['observations']} observations, "
        f"{manifest['counts']['relations']} relations\n"
        f"  Size: {size_kb:.1f} KB"
    )


# ---------- Import ----------

def _read_archive(path: Path) -> tuple[dict, list, list, list]:
    """Read manifest and data lists from a zip archive."""
    with zipfile.ZipFile(path, "r") as zf:
        names = set(zf.namelist())
        required = {"manifest.json", "entities.json", "observations.json", "relations.json"}
        missing = required - names
        if missing:
            raise ValueError(f"Archive missing required files: {sorted(missing)}")

        manifest = json.loads(zf.read("manifest.json"))
        entities = json.loads(zf.read("entities.json"))
        observations = json.loads(zf.read("observations.json"))
        relations = json.loads(zf.read("relations.json"))

    if manifest.get("format_version") not in SUPPORTED_FORMAT_VERSIONS:
        raise ValueError(
            f"Unsupported format_version {manifest.get('format_version')} "
            f"(supported: {sorted(SUPPORTED_FORMAT_VERSIONS)})"
        )
    return manifest, entities, observations, relations


def tool_import_vault(input_path: str, vault: str = "") -> str:
    """Import a vault export zip into a target vault. Always additive.

    Existing entities are matched by name and reused. New observations are
    deduped against existing observations on the matched entity (exact content
    match, superseded rows included). Relations are deduped by
    (from, to, relation_type).

    Superseded observations are imported and re-flagged as superseded once all
    IDs have been remapped, so exported history round-trips intact without old
    facts reappearing as active ones. Soft-deleted rows are still skipped.

    Args:
        input_path: Path to a zip archive produced by export_vault.
        vault: Target vault name. Defaults to the source vault from the
               archive's manifest. Auto-created if it doesn't exist.

    Returns:
        Summary of what was imported, skipped, and reused.
    """
    if not input_path or not input_path.strip():
        return "Error: input_path is required."

    archive = Path(input_path.strip()).expanduser()
    if not archive.exists():
        return f"Error: file not found: {archive}"
    if not archive.is_file():
        return f"Error: not a file: {archive}"

    try:
        manifest, raw_entities, raw_observations, raw_relations = _read_archive(archive)
    except (zipfile.BadZipFile, ValueError, json.JSONDecodeError, KeyError) as e:
        return f"Error reading archive: {e}"

    target_vault = (vault or manifest.get("source_vault", "")).strip()
    if not target_vault:
        return "Error: target vault could not be determined."

    if get_vault(target_vault) is None:
        config_create_vault(target_vault)
        logger.info("Auto-created vault '%s' for import", target_vault)

    store_mod._load_store()

    # --- Entities: merge by name, build old_id -> new_id map ---
    id_map: dict[str, str] = {}
    entities_created = 0
    entities_reused = 0
    entities_skipped_deleted = 0

    for ed in raw_entities:
        if ed.get("deleted"):
            entities_skipped_deleted += 1
            continue
        name = ed.get("name", "").strip()
        etype = ed.get("entity_type", "concept")
        if not name:
            continue

        existing = get_entity_by_name(name, target_vault)
        if existing is not None:
            id_map[ed["id"]] = existing.id
            entities_reused += 1
        else:
            new_ent = create_entity(name, etype, target_vault)
            id_map[ed["id"]] = new_ent.id
            entities_created += 1

    # --- Observations: dedupe by content per entity ---
    #
    # Superseded observations are imported too — dropping them made
    # export -> import lossy (history silently vanished). They are created
    # first like any other row, then re-flagged as superseded in a second pass
    # once every ID has been remapped, so a superseded fact never lands in the
    # target vault as an active one.
    obs_added = 0
    obs_duplicate = 0
    obs_orphan = 0
    obs_skipped_deleted = 0
    obs_superseded = 0
    obs_supersede_declined = 0

    # old_obs_id -> id of the observation now representing it locally (freshly
    # created, or the pre-existing duplicate it deduped onto).
    obs_id_map: dict[str, str] = {}
    # Local rows this import actually created. Only these may be retired in
    # pass 2 — marking a row the target vault already owned would break the
    # module's ADDITIVE contract.
    created_local_ids: set[str] = set()
    # Local rows that some ACTIVE archive row also maps to. If an archive holds
    # two rows with identical content, one active and one superseded, they
    # dedupe onto a single local row; retiring it would hide the active fact.
    active_local_ids: set[str] = set()
    # Per-entity content index, built once (cheaper than a rescan per row) and
    # kept current as we add. Includes superseded rows so re-import is a no-op.
    content_index: dict[str, dict[str, str]] = {}

    def _entity_content_index(new_eid: str) -> dict[str, str]:
        if new_eid not in content_index:
            content_index[new_eid] = {
                o.content: o.id
                for o in get_observations(new_eid, include_superseded=True)
            }
        return content_index[new_eid]

    for od in raw_observations:
        if od.get("deleted"):
            obs_skipped_deleted += 1
            continue

        old_eid = od.get("entity_id", "")
        new_eid = id_map.get(old_eid)
        if new_eid is None:
            obs_orphan += 1
            continue

        content = od.get("content", "")
        if not content:
            continue

        by_content = _entity_content_index(new_eid)
        if content in by_content:
            obs_duplicate += 1
            local_id = by_content[content]
            obs_id_map[od["id"]] = local_id
            if not od.get("superseded_by"):
                active_local_ids.add(local_id)
            continue

        result = add_observation(
            new_eid, content,
            source=od.get("source", ""),
            occurred_at=od.get("occurred_at") or "",
        )
        if result is not None:
            obs_added += 1
            obs_id_map[od["id"]] = result.id
            created_local_ids.add(result.id)
            by_content[content] = result.id
            if not od.get("superseded_by"):
                active_local_ids.add(result.id)

    # Second pass: rebuild supersede chains with remapped IDs.
    for od in raw_observations:
        if od.get("deleted"):
            continue
        old_superseded_by = od.get("superseded_by")
        if not old_superseded_by:
            continue

        local_id = obs_id_map.get(od["id"])
        if local_id is None:
            continue

        # Import is ADDITIVE: never retire a row this import did not create.
        # Without this guard an archive's superseded row could dedupe onto a
        # pre-existing ACTIVE fact in the target vault and silently hide it.
        if local_id not in created_local_ids:
            obs_supersede_declined += 1
            continue

        # Nor retire a row that an ACTIVE archive row also deduped onto — the
        # active fact wins, exactly as it would if the rows were distinct.
        if local_id in active_local_ids:
            obs_supersede_declined += 1
            continue

        # Remap the pointer. If the replacement wasn't in the archive we keep
        # the original (dangling) ID: the row still reads as superseded, which
        # is what matters — it must not resurface as an active fact.
        new_pointer = obs_id_map.get(old_superseded_by)
        if new_pointer is None:
            new_pointer = old_superseded_by
            logger.warning(
                "Import: observation %s superseded by %s, which is not in the "
                "archive — marking superseded with an unresolved pointer.",
                local_id, old_superseded_by,
            )
        if new_pointer == local_id:
            # Self-reference would make the row permanently unreachable
            # without conveying anything; leave it active instead.
            continue

        # Preserve the archive's supersession timestamp. A v1 archive (or a
        # row superseded before the field existed) has none, and the row is
        # then left unstamped so point_in_time keeps inferring the cutover
        # from the replacement's created_at — stamping the import time would
        # make every restored supersession look like it happened today.
        if mark_superseded(local_id, new_pointer,
                           superseded_at=od.get("superseded_at") or ""):
            obs_superseded += 1

    # --- Relations: dedupe by (from, to, type) ---
    rels_added = 0
    rels_duplicate = 0
    rels_orphan = 0

    # Pre-compute existing relation signatures for entities we touched
    touched_new_ids = set(id_map.values())
    existing_signatures = set()
    for eid in touched_new_ids:
        for rel in get_relations_for_entity(eid):
            existing_signatures.add((rel.from_entity, rel.to_entity, rel.relation_type))

    for rd in raw_relations:
        old_from = rd.get("from_entity", "")
        old_to = rd.get("to_entity", "")
        new_from = id_map.get(old_from)
        new_to = id_map.get(old_to)
        if new_from is None or new_to is None:
            rels_orphan += 1
            continue

        rtype = rd.get("relation_type", "related_to")
        signature = (new_from, new_to, rtype)
        if signature in existing_signatures:
            rels_duplicate += 1
            continue

        rel = Relation(
            id="",
            from_entity=new_from,
            to_entity=new_to,
            relation_type=rtype,
            weight=rd.get("weight", 1.0),
            context=rd.get("context", ""),
        )
        # generate fresh ID
        import uuid
        rel.id = uuid.uuid4().hex[:12]
        rel.created_at = datetime.now(timezone.utc).isoformat()
        add_relation(rel)
        existing_signatures.add(signature)
        rels_added += 1

    source_vault = manifest.get("source_vault", "?")
    logger.info(
        "Imported %s -> '%s': +%d entities, +%d observations, +%d relations",
        archive.name, target_vault, entities_created, obs_added, rels_added,
    )

    lines = [
        f"Imported '{archive.name}' (source: '{source_vault}') into vault '{target_vault}':",
        f"  Entities: {entities_created} created, {entities_reused} reused"
        + (f", {entities_skipped_deleted} skipped (deleted)" if entities_skipped_deleted else ""),
        f"  Observations: {obs_added} added, {obs_duplicate} duplicate"
        + (f", {obs_superseded} marked superseded" if obs_superseded else "")
        + (f", {obs_supersede_declined} kept active (already present)"
           if obs_supersede_declined else "")
        + (f", {obs_orphan} orphan" if obs_orphan else "")
        + (f", {obs_skipped_deleted} skipped (deleted)" if obs_skipped_deleted else ""),
        f"  Relations: {rels_added} added, {rels_duplicate} duplicate"
        + (f", {rels_orphan} orphan" if rels_orphan else ""),
    ]
    return "\n".join(lines)
