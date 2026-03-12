"""Entity data model — named nodes in the knowledge graph."""

from dataclasses import dataclass, field
from datetime import datetime, timezone


ENTITY_TYPES = frozenset({
    "person", "project", "concept", "decision", "error", "solution",
    "technology", "event", "organization", "location", "process", "artifact",
})


@dataclass
class Entity:
    """A named node in the knowledge graph."""
    id: str
    name: str
    entity_type: str
    vault: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    deleted: bool = False

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "entity_type": self.entity_type,
            "vault": self.vault,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "deleted": self.deleted,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Entity":
        return cls(
            id=d["id"],
            name=d["name"],
            entity_type=d["entity_type"],
            vault=d["vault"],
            created_at=d.get("created_at", ""),
            updated_at=d.get("updated_at", ""),
            deleted=d.get("deleted", False),
        )
