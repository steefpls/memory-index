"""SQLite persistence layer for entities, observations, relations, and vaults.

This module is the single durable store behind the in-memory caches in
`src.indexer.store` and `src.graph.manager`. Every mutation is a row-level
transaction (no more whole-file rewrites), and SQLite's file locking makes a
concurrent writer from another process fail loudly instead of silently
clobbering the store.

On first open, if the database file does not exist but the legacy JSON store
files do, their contents are migrated in. The JSON files are left untouched
afterwards as a frozen backup — they are never written again.

All functions lazily resolve `src.config.DB_FILE` at call time so tests can
repoint the database at a temp directory; call `reset()` between tests to
close the cached connection.
"""

import json
import logging
import sqlite3
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

_conn: sqlite3.Connection | None = None
_conn_path: Path | None = None
_DB_LOCK = threading.RLock()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS vaults (
    name            TEXT PRIMARY KEY,
    collection_name TEXT NOT NULL,
    created_at      TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS entities (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    vault       TEXT NOT NULL,
    created_at  TEXT NOT NULL DEFAULT '',
    updated_at  TEXT NOT NULL DEFAULT '',
    deleted     INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_entities_vault_name ON entities(vault, name);
CREATE TABLE IF NOT EXISTS observations (
    id            TEXT PRIMARY KEY,
    entity_id     TEXT NOT NULL,
    content       TEXT NOT NULL,
    source        TEXT NOT NULL DEFAULT '',
    created_at    TEXT NOT NULL DEFAULT '',
    occurred_at   TEXT,
    deleted       INTEGER NOT NULL DEFAULT 0,
    superseded_by TEXT NOT NULL DEFAULT '',
    superseded_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_observations_entity ON observations(entity_id);
CREATE TABLE IF NOT EXISTS relations (
    id            TEXT PRIMARY KEY,
    from_entity   TEXT NOT NULL,
    to_entity     TEXT NOT NULL,
    relation_type TEXT NOT NULL,
    weight        REAL NOT NULL DEFAULT 1.0,
    context       TEXT NOT NULL DEFAULT '',
    created_at    TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_relations_from ON relations(from_entity);
CREATE INDEX IF NOT EXISTS idx_relations_to ON relations(to_entity);
"""


def _db_path() -> Path:
    from src import config
    return Path(config.DB_FILE)


def _connect() -> sqlite3.Connection:
    """Open (or return) the singleton connection, creating schema on demand.

    check_same_thread=False because writers run from HTTP handler threads and
    the auto-librarian thread; every use is serialized under _DB_LOCK.
    """
    global _conn, _conn_path
    with _DB_LOCK:
        path = _db_path()
        if _conn is not None and _conn_path == path:
            return _conn
        if _conn is not None:
            try:
                _conn.close()
            except Exception:
                pass
            _conn = None

        path.parent.mkdir(parents=True, exist_ok=True)
        fresh = not path.exists()
        conn = sqlite3.connect(str(path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.executescript(_SCHEMA)
        cur = conn.execute("SELECT value FROM meta WHERE key='schema_version'")
        row = cur.fetchone()
        if row is None:
            conn.execute(
                "INSERT INTO meta(key, value) VALUES ('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )
        conn.commit()

        _conn = conn
        _conn_path = path
        if fresh:
            _migrate_legacy_json(conn)
        return _conn


def reset() -> None:
    """Close the cached connection (tests repoint DB_FILE between cases)."""
    global _conn, _conn_path
    with _DB_LOCK:
        if _conn is not None:
            try:
                _conn.close()
            except Exception:
                pass
        _conn = None
        _conn_path = None


# ---------- Legacy JSON migration ----------

def _migrate_legacy_json(conn: sqlite3.Connection) -> None:
    """One-time import of the legacy JSON store into a freshly created DB.

    Runs only when the DB file was just created. Reads whatever legacy files
    exist (vaults.json, memory_entities.json, memory_graph.json) and inserts
    their rows verbatim — no interpretation, no cleanup. The JSON files are
    deliberately left in place as a backup.
    """
    from src import config

    counts = {"vaults": 0, "entities": 0, "observations": 0, "relations": 0}

    def _read(path: Path):
        try:
            if path.exists():
                return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Legacy migration: failed to read %s: %s", path, e)
        return None

    with _DB_LOCK:
        vaults_data = _read(Path(config.VAULTS_FILE))
        if vaults_data:
            for name, cfg in (vaults_data.get("vaults") or {}).items():
                conn.execute(
                    "INSERT OR REPLACE INTO vaults(name, collection_name, created_at) "
                    "VALUES (?, ?, ?)",
                    (name,
                     cfg.get("collection_name") or f"memory_{name}",
                     cfg.get("created_at", "")),
                )
                counts["vaults"] += 1

        entities_data = _read(Path(config.ENTITIES_FILE))
        if entities_data:
            for ed in entities_data.get("entities", []):
                conn.execute(
                    "INSERT OR REPLACE INTO entities"
                    "(id, name, entity_type, vault, created_at, updated_at, deleted) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (ed["id"], ed["name"], ed["entity_type"], ed["vault"],
                     ed.get("created_at", ""), ed.get("updated_at", ""),
                     1 if ed.get("deleted") else 0),
                )
                counts["entities"] += 1
            for od in entities_data.get("observations", []):
                conn.execute(
                    "INSERT OR REPLACE INTO observations"
                    "(id, entity_id, content, source, created_at, occurred_at, "
                    " deleted, superseded_by, superseded_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (od["id"], od["entity_id"], od["content"],
                     od.get("source", ""), od.get("created_at", ""),
                     od.get("occurred_at") or None,
                     1 if od.get("deleted") else 0,
                     od.get("superseded_by", ""),
                     od.get("superseded_at") or None),
                )
                counts["observations"] += 1

        graph_data = _read(Path(config.GRAPH_FILE))
        if graph_data:
            for rd in graph_data.get("relations", []):
                conn.execute(
                    "INSERT OR REPLACE INTO relations"
                    "(id, from_entity, to_entity, relation_type, weight, context, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (rd["id"], rd["from_entity"], rd["to_entity"],
                     rd["relation_type"], rd.get("weight", 1.0),
                     rd.get("context", ""), rd.get("created_at", "")),
                )
                counts["relations"] += 1

        conn.commit()

    if any(counts.values()):
        logger.info(
            "Migrated legacy JSON store into %s: %d vaults, %d entities, "
            "%d observations, %d relations (JSON files kept as backup)",
            _db_path(), counts["vaults"], counts["entities"],
            counts["observations"], counts["relations"],
        )


# ---------- Vaults ----------

def get_all_vaults() -> list[dict]:
    with _DB_LOCK:
        conn = _connect()
        rows = conn.execute("SELECT * FROM vaults").fetchall()
        return [dict(r) for r in rows]


def upsert_vault(vault: dict) -> None:
    with _DB_LOCK:
        conn = _connect()
        conn.execute(
            "INSERT OR REPLACE INTO vaults(name, collection_name, created_at) "
            "VALUES (?, ?, ?)",
            (vault["name"], vault["collection_name"], vault.get("created_at", "")),
        )
        conn.commit()


def delete_vault_row(name: str) -> None:
    with _DB_LOCK:
        conn = _connect()
        conn.execute("DELETE FROM vaults WHERE name = ?", (name,))
        conn.commit()


# ---------- Load ----------

def load_entities_observations() -> tuple[list[dict], list[dict]]:
    with _DB_LOCK:
        conn = _connect()
        ents = [dict(r) for r in conn.execute("SELECT * FROM entities").fetchall()]
        obs = [dict(r) for r in conn.execute("SELECT * FROM observations").fetchall()]
    for e in ents:
        e["deleted"] = bool(e["deleted"])
    for o in obs:
        o["deleted"] = bool(o["deleted"])
    return ents, obs


def load_relations() -> list[dict]:
    with _DB_LOCK:
        conn = _connect()
        return [dict(r) for r in conn.execute("SELECT * FROM relations").fetchall()]


# ---------- Entities / observations ----------

def upsert_entities(entities) -> None:
    """Write entity rows. Accepts Entity objects or plain dicts."""
    rows = []
    for e in entities:
        d = e if isinstance(e, dict) else e.to_dict()
        rows.append((d["id"], d["name"], d["entity_type"], d["vault"],
                     d.get("created_at", ""), d.get("updated_at", ""),
                     1 if d.get("deleted") else 0))
    if not rows:
        return
    with _DB_LOCK:
        conn = _connect()
        conn.executemany(
            "INSERT OR REPLACE INTO entities"
            "(id, name, entity_type, vault, created_at, updated_at, deleted) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        conn.commit()


def upsert_observations(observations) -> None:
    """Write observation rows. Accepts Observation objects or plain dicts."""
    rows = []
    for o in observations:
        d = o if isinstance(o, dict) else o.to_dict()
        rows.append((d["id"], d["entity_id"], d["content"],
                     d.get("source", ""), d.get("created_at", ""),
                     d.get("occurred_at") or None,
                     1 if d.get("deleted") else 0,
                     d.get("superseded_by", ""),
                     d.get("superseded_at") or None))
    if not rows:
        return
    with _DB_LOCK:
        conn = _connect()
        conn.executemany(
            "INSERT OR REPLACE INTO observations"
            "(id, entity_id, content, source, created_at, occurred_at, "
            " deleted, superseded_by, superseded_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        conn.commit()


def hard_delete_entities(entity_ids: list[str]) -> None:
    if not entity_ids:
        return
    with _DB_LOCK:
        conn = _connect()
        conn.executemany("DELETE FROM entities WHERE id = ?",
                         [(eid,) for eid in entity_ids])
        conn.commit()


def hard_delete_observations(observation_ids: list[str]) -> None:
    if not observation_ids:
        return
    with _DB_LOCK:
        conn = _connect()
        conn.executemany("DELETE FROM observations WHERE id = ?",
                         [(oid,) for oid in observation_ids])
        conn.commit()


# ---------- Relations ----------

def upsert_relations(relations) -> None:
    """Write relation rows. Accepts Relation objects or plain dicts."""
    rows = []
    for r in relations:
        d = r if isinstance(r, dict) else r.to_dict()
        rows.append((d["id"], d["from_entity"], d["to_entity"],
                     d["relation_type"], d.get("weight", 1.0),
                     d.get("context", ""), d.get("created_at", "")))
    if not rows:
        return
    with _DB_LOCK:
        conn = _connect()
        conn.executemany(
            "INSERT OR REPLACE INTO relations"
            "(id, from_entity, to_entity, relation_type, weight, context, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        conn.commit()


def delete_relation_row(relation_id: str) -> None:
    with _DB_LOCK:
        conn = _connect()
        conn.execute("DELETE FROM relations WHERE id = ?", (relation_id,))
        conn.commit()
