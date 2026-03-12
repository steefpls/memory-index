"""Observation data model — facts attached to entities."""

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class Observation:
    """A fact or piece of knowledge attached to an entity."""
    id: str
    entity_id: str
    content: str
    source: str = ""
    confidence: float = 1.0
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    deleted: bool = False

    @property
    def embedding_text(self) -> str:
        """Text used for embedding. Set externally by store using entity context."""
        return getattr(self, "_embedding_text", self.content)

    @embedding_text.setter
    def embedding_text(self, value: str):
        self._embedding_text = value

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "entity_id": self.entity_id,
            "content": self.content,
            "source": self.source,
            "confidence": self.confidence,
            "created_at": self.created_at,
            "deleted": self.deleted,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Observation":
        return cls(
            id=d["id"],
            entity_id=d["entity_id"],
            content=d["content"],
            source=d.get("source", ""),
            confidence=d.get("confidence", 1.0),
            created_at=d.get("created_at", ""),
            deleted=d.get("deleted", False),
        )
