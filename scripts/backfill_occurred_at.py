"""Backfill `occurred_at` on observations that state their own date.

Many older observations were written as "Steven joined Augmentus on 2024-11-04"
— the event time is right there in the text, but the store only knows the
ingestion time. This script lifts that date into the `occurred_at` field so
temporal queries order by when things HAPPENED, not when they were recorded.

Deliberately conservative: an observation is only touched when its content
contains EXACTLY ONE unambiguous ISO date (YYYY-MM-DD). Zero dates means
nothing to lift; two or more means we can't tell which one is the event time
(e.g. "moved from 2024-01-01 to 2025-01-01"), so we skip rather than guess.
Observations that already have an occurred_at are left alone.

Dry-run is the DEFAULT. Nothing is written unless you pass --apply.

Usage:
    python scripts/backfill_occurred_at.py                    # dry run, all vaults
    python scripts/backfill_occurred_at.py --vault work       # dry run, one vault
    python scripts/backfill_occurred_at.py --vault work --apply
"""

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

# Windows consoles default to cp1252, which chokes on characters like '→'
# that appear in observation previews.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.indexer import store as store_mod

# A bare YYYY-MM-DD, not glued to surrounding digits/dashes. The negative
# lookarounds keep us off things like "2024-11-04-01" or version strings, and
# off the date half of a full ISO datetime (which would need different
# handling than a plain date).
_ISO_DATE_RE = re.compile(r"(?<![\d-])(\d{4}-\d{2}-\d{2})(?![\d-])")


def find_single_iso_date(content: str) -> str | None:
    """Return the sole ISO date in `content`, or None.

    None when there are zero matches, more than one DISTINCT date, or the
    match isn't a real calendar date (e.g. 2024-13-45).
    """
    matches = _ISO_DATE_RE.findall(content or "")
    if not matches:
        return None

    distinct = set(matches)
    if len(distinct) != 1:
        return None

    candidate = matches[0]
    try:
        datetime.strptime(candidate, "%Y-%m-%d")
    except ValueError:
        return None
    return candidate


def _vault_of(obs, entities) -> str | None:
    ent = entities.get(obs.entity_id)
    return ent.vault if ent else None


def backfill(vault: str | None, apply_changes: bool) -> int:
    entities, observations = store_mod.snapshot_store()

    matched: list[tuple[str, str, str]] = []  # (obs_id, date, content)
    skipped_no_date = 0
    skipped_ambiguous = 0
    skipped_has_occurred_at = 0
    skipped_deleted = 0
    out_of_scope = 0

    for obs in observations.values():
        if obs.deleted:
            skipped_deleted += 1
            continue

        obs_vault = _vault_of(obs, entities)
        if vault is not None and obs_vault != vault:
            out_of_scope += 1
            continue

        if getattr(obs, "occurred_at", None):
            skipped_has_occurred_at += 1
            continue

        found = find_single_iso_date(obs.content)
        if found is None:
            # Distinguish "no date at all" from "more than one date".
            if _ISO_DATE_RE.search(obs.content or ""):
                skipped_ambiguous += 1
            else:
                skipped_no_date += 1
            continue

        matched.append((obs.id, found, obs.content))

    scope = f"vault '{vault}'" if vault else "all vaults"
    mode = "APPLY" if apply_changes else "DRY RUN"
    print(f"[{mode}] backfill occurred_at over {scope}")
    print(f"  scanned:                 {len(observations) - out_of_scope}")
    print(f"  matched (single date):   {len(matched)}")
    print(f"  skipped, no ISO date:    {skipped_no_date}")
    print(f"  skipped, ambiguous:      {skipped_ambiguous}")
    print(f"  skipped, already set:    {skipped_has_occurred_at}")
    print(f"  skipped, deleted:        {skipped_deleted}")

    if matched:
        print("\n  Changes:")
        for obs_id, date, content in matched:
            preview = content if len(content) <= 90 else content[:87] + "..."
            print(f"    {obs_id}  {date}  {preview}")

    if not apply_changes:
        if matched:
            print(f"\n  Dry run — nothing written. Re-run with --apply to set "
                  f"occurred_at on {len(matched)} observation(s).")
        return 0

    if not matched:
        print("\n  Nothing to apply.")
        return 0

    written = 0
    with store_mod.STORE_LOCK:
        for obs_id, date, _content in matched:
            obs = store_mod._observations.get(obs_id)
            # Re-check under the lock: the store may have moved since the
            # snapshot was taken.
            if obs is None or obs.deleted or obs.occurred_at:
                continue
            obs.occurred_at = date
            written += 1
        store_mod._save_store()

    print(f"\n  Applied: {written} observation(s) updated on disk.")
    print("  Note: Chroma metadata is refreshed on the next re-embed "
          "(scripts/reembed_all.py) — the JSON store is the source of truth.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Backfill occurred_at from unambiguous ISO dates in observation text.",
    )
    parser.add_argument("--vault", default="",
                        help="Limit to one vault (default: all vaults).")
    parser.add_argument("--dry-run", action="store_true", default=True,
                        help="Report what WOULD change without writing. This is "
                             "the default; the flag just makes it explicit.")
    parser.add_argument("--apply", action="store_true",
                        help="Actually write the changes. Overrides --dry-run.")
    args = parser.parse_args()

    return backfill(args.vault.strip() or None, args.apply)


if __name__ == "__main__":
    sys.exit(main())
