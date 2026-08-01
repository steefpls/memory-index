"""Shared test-environment helpers for the SQLite-backed store."""

from pathlib import Path
from unittest.mock import patch


def patch_sqlite(tmpdir, patches: list):
    """Point the SQLite store at a temp file and reset the cached connection.

    Starts a patch on src.config.DB_FILE, appends it to the caller's cleanup
    list, and resets the db module's connection cache so the next store write
    opens the temp database. Returns the db module; tearDown must call
    .reset() on it again BEFORE removing tmpdir — Windows cannot delete a
    directory holding an open database file.
    """
    import src.indexer.db as db_mod
    p = patch("src.config.DB_FILE", Path(tmpdir) / "memory.db")
    patches.append(p)
    p.start()
    db_mod.reset()
    return db_mod


def close_sqlite():
    """Close the cached SQLite connection (safe to call repeatedly)."""
    import src.indexer.db as db_mod
    db_mod.reset()
