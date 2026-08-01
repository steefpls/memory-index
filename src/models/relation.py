"""Relation data model — directed edges between entities."""

from dataclasses import dataclass, field
from datetime import datetime, timezone


# Closed canonical set — enforced at the write boundary (create_relation).
# One spelling per meaning; synonyms live in RELATION_ALIASES below and are
# canonicalized at write time. `related_to` is the deliberate escape hatch:
# anything that fits nothing else goes there, with the nuance in `context`.
RELATION_TYPES = frozenset({
    # generic
    "related_to",
    # structure & artifacts
    "part_of", "uses", "depends_on", "involves", "applies_to",
    "builds_on", "replaces", "created", "maintains",
    # people & organizations
    "works_at", "worked_at", "works_on", "reports_to", "leads",
    "founded", "funds", "friend_of", "collaborates_with",
    "learned_from", "participated_in",
    # knowledge & causality
    "solves", "caused_by", "contradicts", "blocks",
})

# Synonym → (canonical, flip). flip=True means the alias states the same fact
# in the opposite direction, so from/to must be swapped when canonicalizing
# (e.g. "A created_by B" becomes "B created A").
RELATION_ALIASES: dict[str, tuple[str, bool]] = {
    "created_by": ("created", True),
    "creates": ("created", False),
    "authored": ("created", False),
    "built": ("created", False),
    "maintained_by": ("maintains", True),
    "causes": ("caused_by", True),
    "solved_by": ("solves", True),
    "friends_with": ("friend_of", False),
    "close_friend_of": ("friend_of", False),
    "worked_with": ("collaborates_with", False),
    "works_with": ("collaborates_with", False),
    "collaborated_with": ("collaborates_with", False),
    "member_of": ("part_of", False),
    "belongs_to": ("part_of", False),
    "project_of": ("part_of", False),
    "extends": ("builds_on", False),
    "implements": ("builds_on", False),
    "wraps": ("builds_on", False),
    "employs": ("works_at", True),
    "employed_by": ("works_at", False),
    "led": ("leads", False),
    "founded_by": ("founded", True),
    "funded_by": ("funds", True),
}


def canonicalize_relation_type(relation_type: str) -> tuple[str, bool] | None:
    """Resolve a relation type to its canonical form.

    Returns (canonical_type, flip) — flip=True means from/to must be swapped —
    or None if the type is neither canonical nor a known alias.
    """
    rt = (relation_type or "").strip().lower()
    if rt in RELATION_TYPES:
        return rt, False
    return RELATION_ALIASES.get(rt)


@dataclass
class Relation:
    """A directed edge between two entities in the knowledge graph."""
    id: str
    from_entity: str
    to_entity: str
    relation_type: str
    weight: float = 1.0
    context: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "from_entity": self.from_entity,
            "to_entity": self.to_entity,
            "relation_type": self.relation_type,
            "weight": self.weight,
            "context": self.context,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Relation":
        return cls(
            id=d["id"],
            from_entity=d["from_entity"],
            to_entity=d["to_entity"],
            relation_type=d["relation_type"],
            weight=d.get("weight", 1.0),
            context=d.get("context", ""),
            created_at=d.get("created_at", ""),
        )
