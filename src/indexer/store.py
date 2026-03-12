"""Entity/observation CRUD with ChromaDB vector storage.

Manages the lifecycle of entities and their observations, embedding observation
text into ChromaDB for semantic retrieval.
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

from src.config import DATA_DIR, ENTITIES_FILE, VAULTS, get_vault
from src.indexer.calibration import calibrate_collection
from src.indexer.embedder import get_embedding_function, get_collection, get_chroma_client
from src.models.entity import Entity
from src.models.observation import Observation

_RECALIBRATE_EVERY = 10  # auto-recalibrate after every N observations per vault

logger = logging.getLogger(__name__)

# In-memory entity and observation stores, keyed by entity ID
_entities: dict[str, Entity] = {}
_observations: dict[str, Observation] = {}  # keyed by observation ID
_loaded = False


def _generate_id() -> str:
    return uuid.uuid4().hex[:12]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_store() -> None:
    """Load entities and observations from disk."""
    global _entities, _observations, _loaded
    if _loaded:
        return

    if ENTITIES_FILE.exists():
        try:
            data = json.loads(ENTITIES_FILE.read_text(encoding="utf-8"))
            for ed in data.get("entities", []):
                ent = Entity.from_dict(ed)
                _entities[ent.id] = ent
            for od in data.get("observations", []):
                obs = Observation.from_dict(od)
                _observations[obs.id] = obs
            logger.info("Loaded %d entities, %d observations from disk",
                        len(_entities), len(_observations))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load entity store: %s", e)

    _loaded = True


def _save_store() -> None:
    """Save entities and observations to disk."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    data = {
        "entities": [e.to_dict() for e in _entities.values()],
        "observations": [o.to_dict() for o in _observations.values()],
    }
    ENTITIES_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


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

    # Check for existing entity with same name in vault
    for ent in _entities.values():
        if ent.name == name and ent.vault == vault and not ent.deleted:
            logger.info("Entity '%s' already exists in vault '%s'", name, vault)
            # Add any new observations
            if observations:
                for content in observations:
                    add_observation(ent.id, content, source=source)
            return ent

    entity = Entity(
        id=_generate_id(),
        name=name,
        entity_type=entity_type,
        vault=vault,
    )
    _entities[entity.id] = entity

    # Add initial observations
    if observations:
        for content in observations:
            add_observation(entity.id, content, source=source)

    _save_store()
    logger.info("Created entity: %s (%s) in vault %s", name, entity_type, vault)
    return entity


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
    for ent in _entities.values():
        if ent.name == name and ent.vault == vault and not ent.deleted:
            return ent
    return None


def update_entity(entity_id: str, name: str | None = None,
                  entity_type: str | None = None) -> Entity | None:
    """Update an entity's name or type."""
    _load_store()
    ent = _entities.get(entity_id)
    if ent is None or ent.deleted:
        return None

    if name is not None:
        ent.name = name
    if entity_type is not None:
        ent.entity_type = entity_type
    ent.updated_at = _now_iso()

    _save_store()

    # Re-embed all observations if entity name/type changed
    if name is not None or entity_type is not None:
        _reembed_entity_observations(ent)

    return ent


def delete_entity(entity_id: str) -> bool:
    """Soft delete an entity and its observations."""
    _load_store()
    ent = _entities.get(entity_id)
    if ent is None or ent.deleted:
        return False

    ent.deleted = True
    ent.updated_at = _now_iso()

    # Soft delete all observations
    obs_ids_to_remove = []
    for obs in _observations.values():
        if obs.entity_id == entity_id and not obs.deleted:
            obs.deleted = True
            obs_ids_to_remove.append(obs.id)

    # Remove from ChromaDB
    if obs_ids_to_remove:
        try:
            collection = _get_collection_for_vault(ent.vault)
            collection.delete(ids=obs_ids_to_remove)
        except Exception as e:
            logger.warning("Failed to remove observations from ChromaDB: %s", e)

    _save_store()
    logger.info("Soft deleted entity: %s (%s)", ent.name, entity_id)
    return True


def list_entities(vault: str | None = None, entity_type: str | None = None,
                  offset: int = 0, limit: int = 50) -> tuple[list[Entity], int]:
    """List entities with optional filters. Returns (entities, total_count)."""
    _load_store()
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
    for e in _entities.values():
        if e.name == name_or_id and not e.deleted:
            return e
    return None


# --- Observation CRUD ---

def add_observation(entity_id: str, content: str, source: str = "",
                    confidence: float = 1.0,
                    supersedes: str = "") -> Observation | None:
    """Add an observation to an entity and embed it in ChromaDB.

    Args:
        entity_id: Entity to attach to.
        content: Observation text.
        source: Optional source attribution.
        confidence: Confidence level (0.0 to 1.0).
        supersedes: Optional observation ID that this new observation replaces.
                    The old observation is marked superseded and removed from search.
    """
    _load_store()
    ent = _entities.get(entity_id)
    if ent is None or ent.deleted:
        return None

    obs = Observation(
        id=_generate_id(),
        entity_id=entity_id,
        content=content,
        source=source,
        confidence=confidence,
    )
    _observations[obs.id] = obs

    # Mark the old observation as superseded and tag it in ChromaDB
    if supersedes:
        old_obs = _observations.get(supersedes)
        if old_obs and not old_obs.deleted and old_obs.entity_id == entity_id:
            old_obs.superseded_by = obs.id
            # Update ChromaDB metadata to tag it as superseded (keep it searchable)
            try:
                collection = _get_collection_for_vault(ent.vault)
                collection.update(
                    ids=[supersedes],
                    metadatas=[{
                        "entity_id": entity_id,
                        "entity_name": ent.name,
                        "entity_type": ent.entity_type,
                        "content": old_obs.content,
                        "source": old_obs.source,
                        "confidence": old_obs.confidence,
                        "vault": ent.vault,
                        "created_at": old_obs.created_at,
                        "superseded_by": obs.id,
                    }],
                )
            except Exception as e:
                logger.warning("Failed to tag superseded observation in ChromaDB: %s", e)

    # Embed in ChromaDB
    embed_text = _make_embedding_text(ent, content)
    try:
        collection = _get_collection_for_vault(ent.vault)
        ef = get_embedding_function(role="index")
        embeddings = ef([embed_text])
        collection.add(
            ids=[obs.id],
            embeddings=embeddings,
            documents=[embed_text],
            metadatas=[{
                "entity_id": entity_id,
                "entity_name": ent.name,
                "entity_type": ent.entity_type,
                "content": content,
                "source": source,
                "confidence": confidence,
                "vault": ent.vault,
                "created_at": obs.created_at,
            }],
        )
    except Exception as e:
        logger.error("Failed to embed observation: %s", e)

    # Update entity timestamp
    ent.updated_at = _now_iso()
    _save_store()

    # Auto-recalibrate every N observations
    vault_obs_count = get_observation_count(ent.vault)
    if vault_obs_count > 0 and vault_obs_count % _RECALIBRATE_EVERY == 0:
        try:
            collection = _get_collection_for_vault(ent.vault)
            calibrate_collection(collection, ent.vault)
            from src.tools.search import invalidate_calibration_cache
            invalidate_calibration_cache(ent.vault)
            logger.info("Auto-recalibrated vault '%s' at %d observations",
                        ent.vault, vault_obs_count)
        except Exception as e:
            logger.warning("Auto-recalibration failed for vault '%s': %s",
                           ent.vault, e)

    return obs


def get_observations(entity_id: str, include_superseded: bool = False) -> list[Observation]:
    """Get observations for an entity.

    Args:
        entity_id: Entity ID.
        include_superseded: If False (default), excludes superseded observations.
                            If True, returns all including superseded (for history).
    """
    _load_store()
    return [
        o for o in _observations.values()
        if o.entity_id == entity_id and not o.deleted
        and (include_superseded or not o.is_superseded)
    ]


def delete_observation(observation_id: str) -> bool:
    """Soft delete an observation and remove from ChromaDB."""
    _load_store()
    obs = _observations.get(observation_id)
    if obs is None or obs.deleted:
        return False

    obs.deleted = True
    ent = _entities.get(obs.entity_id)

    # Remove from ChromaDB
    if ent:
        try:
            collection = _get_collection_for_vault(ent.vault)
            collection.delete(ids=[observation_id])
        except Exception as e:
            logger.warning("Failed to remove observation from ChromaDB: %s", e)

    _save_store()
    return True


def _reembed_entity_observations(entity: Entity) -> None:
    """Re-embed all observations for an entity (after name/type change)."""
    obs_list = get_observations(entity.id)
    if not obs_list:
        return

    try:
        collection = _get_collection_for_vault(entity.vault)
        ef = get_embedding_function(role="index")

        ids = []
        texts = []
        metadatas = []
        for obs in obs_list:
            embed_text = _make_embedding_text(entity, obs.content)
            ids.append(obs.id)
            texts.append(embed_text)
            metadatas.append({
                "entity_id": entity.id,
                "entity_name": entity.name,
                "entity_type": entity.entity_type,
                "content": obs.content,
                "source": obs.source,
                "confidence": obs.confidence,
                "vault": entity.vault,
                "created_at": obs.created_at,
            })

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
    return sum(
        1 for e in _entities.values()
        if not e.deleted and (vault is None or e.vault == vault)
    )


def get_observation_count(vault: str | None = None) -> int:
    """Count active (non-deleted, non-superseded) observations."""
    _load_store()
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
