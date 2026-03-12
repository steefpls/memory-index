"""Relation data model — directed edges between entities."""

from dataclasses import dataclass, field
from datetime import datetime, timezone


RELATION_TYPES = frozenset({
    "depends_on", "solves", "related_to", "contradicts", "builds_on",
    "applies_to", "caused_by", "part_of", "uses", "created_by",
    "maintained_by", "replaces", "extends", "implements", "blocks",
})


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
