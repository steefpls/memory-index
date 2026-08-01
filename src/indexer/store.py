"""Entity/observation CRUD with ChromaDB vector storage.

Manages the lifecycle of entities and their observations, embedding observation
text into ChromaDB for semantic retrieval.

The in-memory dicts are the working set; every mutation is persisted as a
row-level SQLite transaction via `src.indexer.db` (single source of truth for
content — Chroma holds only vectors plus the minimal metadata its `where`
filters need).
"""

import json
import logging
import os
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from src.config import VAULTS, get_vault
from src.indexer import db
from src.indexer.calibration import calibrate_collection
from src.indexer.embedder import get_embedding_function, get_collection
from src.models.entity import Entity
from src.models.observation import Observation

_RECALIBRATE_EVERY = 10  # auto-recalibrate after every N observations per vault
_LIBRARIAN_EVERY = 10    # auto-run librarian after every N observations per vault

logger = logging.getLogger(__name__)

# In-memory entity and observation stores, keyed by entity ID
_entities: dict[str, Entity] = {}
_observations: dict[str, Observation] = {}  # keyed by observation ID
_loaded = False

# Guards every mutation, save, and full-dict scan of _entities/_observations.
# Re-entrant so nested store calls (e.g. create_entity -> add_observation)
# don't self-deadlock. Background workers must snapshot under this lock and
# do their computation outside it.
STORE_LOCK = threading.RLock()


def _generate_id() -> str:
    return uuid.uuid4().hex[:12]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, data, *, indent: int = 2) -> None:
    """Serialize `data` to `path` crash-safely.

    Writes a temp file in the SAME directory (so os.replace stays on one
    volume — required for atomicity on Windows), flushes + fsyncs it, then
    atomically swaps it into place. A crash mid-write leaves either the old
    file or the new one, never a truncated hybrid.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, indent=indent)

    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _load_store() -> None:
    """Load entities and observations from SQLite (migrating legacy JSON once)."""
    global _entities, _observations, _loaded
    with STORE_LOCK:
        if _loaded:
            return

        try:
            ent_rows, obs_rows = db.load_entities_observations()
            for ed in ent_rows:
                ent = Entity.from_dict(ed)
                _entities[ent.id] = ent
            for od in obs_rows:
                obs = Observation.from_dict(od)
                _observations[obs.id] = obs
            logger.info("Loaded %d entities, %d observations from SQLite",
                        len(_entities), len(_observations))
        except Exception as e:
            logger.warning("Failed to load entity store: %s", e)

        _loaded = True


def snapshot_store() -> tuple[dict[str, Entity], dict[str, Observation]]:
    """Return shallow copies of the store dicts, taken under the lock.

    Background workers use this to get a stable view they can iterate without
    holding the lock (and without risking "dict changed size during iteration").
    """
    _load_store()
    with STORE_LOCK:
        return dict(_entities), dict(_observations)


def _get_collection_for_vault(vault_name: str):
    """Get or create the ChromaDB collection for a vault."""
    vault = get_vault(vault_name)
    if vault is None:
        raise ValueError(f"Vault '{vault_name}' does not exist")
    return get_collection(vault.collection_name)


def _make_embedding_text(entity: Entity, content: str) -> str:
    """Build embedding text that includes entity context for better retrieval.

    Pattern mirrors code-index's namespace/class prefix approach:
    "{entity_type}: {entity_name}\n{content}"
    """
    return f"{entity.entity_type}: {entity.name}\n{content}"


# --- Entity CRUD ---

def create_entity(name: str, entity_type: str, vault: str,
                  observations: list[str] | None = None,
                  source: str = "") -> Entity:
    """Create an entity with optional initial observations."""
    _load_store()

    # The find-or-create step must be atomic so two concurrent callers can't
    # both create the same named entity.
    with STORE_LOCK:
        target = None
        for ent in _entities.values():
            if ent.name == name and ent.vault == vault and not ent.deleted:
                target = ent
                break

        if target is not None:
            logger.info("Entity '%s' already exists in vault '%s'", name, vault)
        else:
            target = Entity(
                id=_generate_id(),
                name=name,
                entity_type=entity_type,
                vault=vault,
            )
            _entities[target.id] = target
            db.upsert_entities([target])
            logger.info("Created entity: %s (%s) in vault %s",
                        name, entity_type, vault)

    # Initial observations go through the batch path: one embedder call,
    # one Chroma add, one save — regardless of how many facts there are.
    if observations:
        add_observations(target.id, observations, source=source)

    return target


def get_entity(entity_id: str) -> Entity | None:
    """Get an entity by ID."""
    _load_store()
    ent = _entities.get(entity_id)
    if ent and not ent.deleted:
        return ent
    return None


def get_entity_by_name(name: str, vault: str) -> Entity | None:
    """Get an entity by name within a vault."""
    _load_store()
    with STORE_LOCK:
        for ent in _entities.values():
            if ent.name == name and ent.vault == vault and not ent.deleted:
                return ent
    return None


def update_entity(entity_id: str, name: str | None = None,
                  entity_type: str | None = None) -> Entity | None:
    """Update an entity's name or type."""
    _load_store()
    with STORE_LOCK:
        ent = _entities.get(entity_id)
        if ent is None or ent.deleted:
            return None

        if name is not None:
            ent.name = name
        if entity_type is not None:
            ent.entity_type = entity_type
        ent.updated_at = _now_iso()

        db.upsert_entities([ent])

    # Re-embed all observations if entity name/type changed. Done outside the
    # lock — it's an embedder round-trip, not a store mutation.
    if name is not None or entity_type is not None:
        _reembed_entity_observations(ent)

    return ent


def delete_entity(entity_id: str) -> bool:
    """Soft delete an entity and its observations.

    Also hard-removes any relations involving this entity from the graph,
    so the graph never accumulates dangling references to deleted entities.
    """
    _load_store()
    with STORE_LOCK:
        ent = _entities.get(entity_id)
        if ent is None or ent.deleted:
            return False

        ent.deleted = True
        ent.updated_at = _now_iso()

        # Soft delete all observations
        obs_ids_to_remove = []
        touched_obs = []
        for obs in _observations.values():
            if obs.entity_id == entity_id and not obs.deleted:
                obs.deleted = True
                obs_ids_to_remove.append(obs.id)
                touched_obs.append(obs)

        db.upsert_entities([ent])
        db.upsert_observations(touched_obs)

    # Remove from ChromaDB
    if obs_ids_to_remove:
        try:
            collection = _get_collection_for_vault(ent.vault)
            collection.delete(ids=obs_ids_to_remove)
        except Exception as e:
            logger.warning("Failed to remove observations from ChromaDB: %s", e)

    # Always clean up relations so the graph can't accumulate dangling edges.
    # Imported here (not at module top) to avoid a circular import with graph.manager.
    try:
        from src.graph.manager import remove_entity_relations
        removed = remove_entity_relations(entity_id)
        if removed:
            logger.info("Removed %d relations for deleted entity %s", removed, entity_id)
    except Exception as e:
        logger.warning("Failed to remove relations for entity %s: %s", entity_id, e)

    logger.info("Soft deleted entity: %s (%s)", ent.name, entity_id)
    return True


def list_entities(vault: str | None = None, entity_type: str | None = None,
                  offset: int = 0, limit: int = 50) -> tuple[list[Entity], int]:
    """List entities with optional filters. Returns (entities, total_count)."""
    _load_store()
    with STORE_LOCK:
        filtered = [
            e for e in _entities.values()
            if not e.deleted
            and (vault is None or e.vault == vault)
            and (entity_type is None or e.entity_type == entity_type)
        ]
    filtered.sort(key=lambda e: e.updated_at, reverse=True)
    total = len(filtered)
    return filtered[offset:offset + limit], total


def resolve_entity(name_or_id: str, vault: str | None = None) -> Entity | None:
    """Resolve an entity by name or ID. Name resolution requires vault."""
    _load_store()
    # Try ID first
    ent = get_entity(name_or_id)
    if ent is not None:
        return ent
    # Try name (requires vault)
    if vault:
        return get_entity_by_name(name_or_id, vault)
    # Try name across all vaults (return first match)
    with STORE_LOCK:
        for e in _entities.values():
            if e.name == name_or_id and not e.deleted:
                return e
    return None


# --- Observation CRUD ---

def _obs_metadata(ent: Entity, obs: Observation) -> dict:
    """Build the Chroma metadata dict for an observation.

    Deliberately minimal: only the keys Chroma `where` filters need
    (entity_type equality, entity_id $in for the associative pass). Content,
    source, timestamps, and supersession live in the SQLite store — search
    joins back to it by observation id, so the two stores can no longer hold
    divergent copies of a fact.
    """
    return {
        "entity_id": ent.id,
        "entity_type": ent.entity_type,
    }


def _crossed_multiple(prev_count: int, new_count: int, every: int) -> bool:
    """True when the count moved past a multiple of `every` on this write.

    A modulo test (`count % every == 0`) only fires when the count lands
    *exactly* on a boundary, which a batched write of size > 1 can step clean
    over — after which the trigger is unreachable forever. Testing whether the
    boundary was CROSSED fires exactly once per boundary, batch or not.
    """
    if every <= 0 or new_count <= 0:
        return False
    return (new_count // every) > (max(prev_count, 0) // every)


def _run_post_write_hooks(vault: str, vault_obs_count: int,
                          prev_obs_count: int = -1) -> None:
    """Auto-recalibrate / auto-librarian triggers. Call OUTSIDE the store lock.

    Both are expensive (embedding + clustering) and must not be run while
    holding STORE_LOCK, or every reader stalls behind them.

    `prev_obs_count` is the vault's observation count *before* this write.
    Defaults to `vault_obs_count - 1` so a single-observation caller behaves as
    it always did.
    """
    if vault_obs_count <= 0:
        return
    if prev_obs_count < 0:
        prev_obs_count = vault_obs_count - 1

    if _crossed_multiple(prev_obs_count, vault_obs_count, _RECALIBRATE_EVERY):
        try:
            collection = _get_collection_for_vault(vault)
            calibrate_collection(collection, vault)
            from src.tools.search import invalidate_calibration_cache
            invalidate_calibration_cache(vault)
            logger.info("Auto-recalibrated vault '%s' at %d observations",
                        vault, vault_obs_count)
        except Exception as e:
            logger.warning("Auto-recalibration failed for vault '%s': %s", vault, e)

    if _crossed_multiple(prev_obs_count, vault_obs_count, _LIBRARIAN_EVERY):
        def _bg_librarian(vault_name: str):
            try:
                # Snapshot the store under the lock so the librarian works from
                # a stable view, then release before doing any computation.
                with STORE_LOCK:
                    snapshot_entities, snapshot_obs = snapshot_store()
                logger.debug(
                    "Auto-librarian snapshot for '%s': %d entities, %d observations",
                    vault_name, len(snapshot_entities), len(snapshot_obs),
                )
                from src.tools.librarian import tool_run_librarian
                report = tool_run_librarian(vault_name)
                logger.info("Auto-librarian for vault '%s' at %d observations:\n%s",
                            vault_name, vault_obs_count, report)
            except Exception as e:
                logger.warning("Auto-librarian failed for vault '%s': %s",
                               vault_name, e)

        threading.Thread(
            target=_bg_librarian, args=(vault,),
            daemon=True, name="memory-index-auto-librarian",
        ).start()


def add_observation(entity_id: str, content: str, source: str = "",
                    supersedes: str = "",
                    occurred_at: str = "") -> Observation | None:
    """Add an observation to an entity and embed it in ChromaDB.

    Args:
        entity_id: Entity to attach to.
        content: Observation text.
        source: Optional source attribution.
        supersedes: Optional observation ID that this new observation replaces.
                    The old observation is marked superseded and removed from search.
        occurred_at: Optional ISO date/datetime for when the fact actually
                     happened (event time), as opposed to created_at which is
                     when it was recorded (ingestion time).
    """
    _load_store()

    with STORE_LOCK:
        ent = _entities.get(entity_id)
        if ent is None or ent.deleted:
            return None

        # Captured before the write so the post-write hooks can tell whether
        # this call CROSSED a trigger boundary (see _crossed_multiple).
        prev_obs_count = get_observation_count(ent.vault)

        obs = Observation(
            id=_generate_id(),
            entity_id=entity_id,
            content=content,
            source=source,
            occurred_at=(occurred_at or "").strip() or None,
        )
        _observations[obs.id] = obs

        # Mark the old observation as superseded. superseded_at records WHEN
        # the supersession was applied, turning the pointer into a validity
        # interval (believed from created_at until superseded_at).
        touched = [obs]
        if supersedes:
            old_obs = _observations.get(supersedes)
            if old_obs and not old_obs.deleted and old_obs.entity_id == entity_id:
                old_obs.superseded_by = obs.id
                old_obs.superseded_at = _now_iso()
                touched.append(old_obs)

        ent.updated_at = _now_iso()
        db.upsert_observations(touched)
        db.upsert_entities([ent])
        vault = ent.vault
        embed_text = _make_embedding_text(ent, content)
        add_meta = _obs_metadata(ent, obs)
        vault_obs_count = get_observation_count(vault)

    # --- Chroma I/O outside the lock ---
    try:
        collection = _get_collection_for_vault(vault)
        ef = get_embedding_function()
        embeddings = ef([embed_text])
        collection.add(
            ids=[obs.id],
            embeddings=embeddings,
            documents=[embed_text],
            metadatas=[add_meta],
        )
    except Exception as e:
        logger.error("Failed to embed observation: %s", e)

    _run_post_write_hooks(vault, vault_obs_count, prev_obs_count)

    return obs


def add_observations(entity_id: str, contents: list[str], source: str = "",
                     occurred_at: list[str] | None = None) -> list[Observation]:
    """Add several observations to one entity in a single batch.

    Each content becomes its own Observation with its own embedding (atomicity
    of facts is preserved), but the whole batch costs ONE embedder call, ONE
    Chroma add, and ONE store save instead of N of each.

    Args:
        entity_id: Entity to attach to.
        contents: List of observation texts. Empty/whitespace items are dropped.
        source: Source attribution applied to every observation in the batch.
        occurred_at: Optional parallel list of ISO date/datetime strings (event
                     time). Must be the same length as `contents`, or omitted.

    Returns:
        The created Observations, in input order. Empty list if the entity is
        unknown/deleted or no non-empty contents were supplied.

    Raises:
        ValueError: if `occurred_at` is provided with a different length than
                    `contents`.
    """
    _load_store()

    if contents is None:
        return []
    if occurred_at is not None and len(occurred_at) != len(contents):
        raise ValueError(
            f"occurred_at has {len(occurred_at)} entries but contents has "
            f"{len(contents)} — they must match (or omit occurred_at)."
        )

    # Keep contents and their event times paired while filtering blanks.
    pairs: list[tuple[str, str | None]] = []
    for i, raw in enumerate(contents):
        text = (raw or "").strip()
        if not text:
            continue
        when = None
        if occurred_at is not None:
            when = (occurred_at[i] or "").strip() or None
        pairs.append((text, when))

    if not pairs:
        return []

    with STORE_LOCK:
        ent = _entities.get(entity_id)
        if ent is None or ent.deleted:
            return []

        # See add_observation: the hooks trigger on boundary CROSSING, which a
        # batch of size > 1 would otherwise step straight over.
        prev_obs_count = get_observation_count(ent.vault)

        created: list[Observation] = []
        for text, when in pairs:
            obs = Observation(
                id=_generate_id(),
                entity_id=entity_id,
                content=text,
                source=source,
                occurred_at=when,
            )
            _observations[obs.id] = obs
            created.append(obs)

        ent.updated_at = _now_iso()
        db.upsert_observations(created)
        db.upsert_entities([ent])

        vault = ent.vault
        ids = [o.id for o in created]
        texts = [_make_embedding_text(ent, o.content) for o in created]
        metadatas = [_obs_metadata(ent, o) for o in created]
        vault_obs_count = get_observation_count(vault)

    # --- One embedder call, one Chroma add, outside the lock ---
    try:
        collection = _get_collection_for_vault(vault)
        ef = get_embedding_function()
        embeddings = ef(texts)
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
    except Exception as e:
        logger.error("Failed to embed observation batch (%d items): %s",
                     len(ids), e)

    _run_post_write_hooks(vault, vault_obs_count, prev_obs_count)

    return created


def mark_superseded(observation_id: str, superseded_by: str,
                    superseded_at: str = "") -> bool:
    """Flag an existing observation as superseded by another one.

    Unlike add_observation(supersedes=...), this does not create anything — it
    only records the pointer on an observation that already exists. Import uses
    it to rebuild supersede chains after all rows have been re-created, so a
    superseded fact can never land in the target vault as an active one.

    `superseded_by` may be an ID that isn't present locally (a chain whose head
    was not in the archive); the row is still marked superseded, which is what
    keeps it out of active reads and search.

    `superseded_at` carries the ORIGINAL supersession timestamp. When it is
    empty the row is left unstamped (None) rather than stamped with the
    current time: this is a reconstruction primitive, so "now" is never the
    truthful answer — an archive row without a stamp was superseded at some
    unknown past moment, and None is what makes `point_in_time` fall back to
    inferring the cutover from the replacement's created_at. Stamping now
    would collapse every restored supersession onto the restore date and
    silently rewrite history.
    """
    _load_store()
    with STORE_LOCK:
        obs = _observations.get(observation_id)
        if obs is None or obs.deleted or not superseded_by:
            return False
        obs.superseded_by = superseded_by
        obs.superseded_at = (superseded_at or "").strip() or None
        db.upsert_observations([obs])
    return True


def get_observation(observation_id: str) -> Observation | None:
    """Get a single non-deleted observation by ID (superseded rows included).

    Search uses this to join Chroma hits back to the authoritative store row —
    a vector whose observation is gone (deleted or diverged) resolves to None
    and is dropped from results.
    """
    _load_store()
    with STORE_LOCK:
        obs = _observations.get(observation_id)
        if obs is None or obs.deleted:
            return None
        return obs


def get_observations(entity_id: str, include_superseded: bool = False) -> list[Observation]:
    """Get observations for an entity.

    Args:
        entity_id: Entity ID.
        include_superseded: If False (default), excludes superseded observations.
                            If True, returns all including superseded (for history).
    """
    _load_store()
    with STORE_LOCK:
        return [
            o for o in _observations.values()
            if o.entity_id == entity_id and not o.deleted
            and (include_superseded or not o.is_superseded)
        ]


def delete_observation(observation_id: str) -> bool:
    """Soft delete an observation and remove from ChromaDB."""
    _load_store()
    with STORE_LOCK:
        obs = _observations.get(observation_id)
        if obs is None or obs.deleted:
            return False

        obs.deleted = True
        ent = _entities.get(obs.entity_id)
        vault = ent.vault if ent else None
        db.upsert_observations([obs])

    # Remove from ChromaDB
    if vault is not None:
        try:
            collection = _get_collection_for_vault(vault)
            collection.delete(ids=[observation_id])
        except Exception as e:
            logger.warning("Failed to remove observation from ChromaDB: %s", e)

    return True


def _reembed_entity_observations(entity: Entity) -> None:
    """Re-embed all observations for an entity (after name/type change)."""
    obs_list = get_observations(entity.id)
    if not obs_list:
        return

    try:
        collection = _get_collection_for_vault(entity.vault)
        ef = get_embedding_function()

        ids = []
        texts = []
        metadatas = []
        for obs in obs_list:
            embed_text = _make_embedding_text(entity, obs.content)
            ids.append(obs.id)
            texts.append(embed_text)
            metadatas.append(_obs_metadata(entity, obs))

        embeddings = ef(texts)
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
    except Exception as e:
        logger.error("Failed to re-embed observations for entity %s: %s", entity.id, e)


# --- Stats ---

def get_entity_count(vault: str | None = None) -> int:
    """Count active entities."""
    _load_store()
    with STORE_LOCK:
        return sum(
            1 for e in _entities.values()
            if not e.deleted and (vault is None or e.vault == vault)
        )


def get_observation_count(vault: str | None = None) -> int:
    """Count active (non-deleted, non-superseded) observations."""
    _load_store()
    with STORE_LOCK:
        if vault is None:
            return sum(1 for o in _observations.values()
                       if not o.deleted and not o.is_superseded)
        vault_entity_ids = {
            e.id for e in _entities.values()
            if not e.deleted and e.vault == vault
        }
        return sum(
            1 for o in _observations.values()
            if not o.deleted and not o.is_superseded
            and o.entity_id in vault_entity_ids
        )
