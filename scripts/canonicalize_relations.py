"""One-time cleanup: collapse legacy relation types onto the canonical set.

The write boundary now enforces `RELATION_TYPES` (with `RELATION_ALIASES`
auto-canonicalized), but rows written before enforcement carry ~40 ad-hoc
types. This script maps every non-canonical relation to its canonical form:

  1. Known aliases resolve exactly as the write boundary would — including
     direction flips (created_by -> created with endpoints swapped).
  2. Legacy one-off types with no canonical equivalent become `related_to`.
  3. In both cases the original type is preserved by prefixing the relation's
     context with "[was: <original>]", so no semantics are lost.

Dry-run by default; pass --apply to write. Run with the service STOPPED —
this bypasses the daemon and writes to the SQLite store directly.

Usage:
    python scripts/canonicalize_relations.py            # preview
    python scripts/canonicalize_relations.py --apply    # execute
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.graph.manager import get_all_relations, remove_relation, add_relation
from src.models.relation import (
    Relation, RELATION_TYPES, canonicalize_relation_type,
)

# Legacy types seen in the live vault that have no general alias. Everything
# not listed here and not resolvable via canonicalize_relation_type falls back
# to related_to as well — this map only exists for the handful of legacy types
# with a better target than the generic fallback.
LEGACY_MAP: dict[str, tuple[str, bool]] = {
    "friend_and_ex_colleague": ("friend_of", False),
}


def resolve(relation_type: str) -> tuple[str, bool]:
    """Resolve any type to (canonical, flip), falling back to related_to."""
    hit = canonicalize_relation_type(relation_type)
    if hit is not None:
        return hit
    hit = LEGACY_MAP.get(relation_type.strip().lower())
    if hit is not None:
        return hit
    return "related_to", False


def main() -> int:
    apply = "--apply" in sys.argv

    relations = get_all_relations()
    plan = []
    for rel in relations:
        if rel.relation_type in RELATION_TYPES:
            continue
        canonical, flip = resolve(rel.relation_type)
        plan.append((rel, canonical, flip))

    if not plan:
        print(f"All {len(relations)} relations already canonical. Nothing to do.")
        return 0

    print(f"{len(plan)} of {len(relations)} relations need canonicalizing:\n")
    for rel, canonical, flip in plan:
        arrow = "<-flip->" if flip else "->"
        print(f"  {rel.id}: {rel.relation_type} {arrow} {canonical}")

    if not apply:
        print("\n(dry run — pass --apply to execute)")
        return 0

    changed = 0
    for rel, canonical, flip in plan:
        original = rel.relation_type
        from_e, to_e = rel.from_entity, rel.to_entity
        if flip:
            from_e, to_e = to_e, from_e
        context = f"[was: {original}]"
        if rel.context:
            context += f" {rel.context}"

        # remove + re-add under the same id so the graph edge endpoints and
        # the SQLite row both reflect the flip.
        remove_relation(rel.id)
        add_relation(Relation(
            id=rel.id,
            from_entity=from_e,
            to_entity=to_e,
            relation_type=canonical,
            weight=rel.weight,
            context=context,
            created_at=rel.created_at,
        ))
        changed += 1

    print(f"\nCanonicalized {changed} relations.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
