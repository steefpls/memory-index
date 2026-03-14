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
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    deleted: bool = False
    superseded_by: str = ""  # ID of the observation that replaced this one

    @property
    def embedding_text(self) -> str:
        """Text used for embedding. Set externally by store using entity context."""
        return getattr(self, "_embedding_text", self.content)

    @embedding_text.setter
    def embedding_text(self, value: str):
        self._embedding_text = value

    @property
    def is_superseded(self) -> bool:
        return bool(self.superseded_by)

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "entity_id": self.entity_id,
            "content": self.content,
            "source": self.source,
            "created_at": self.created_at,
            "deleted": self.deleted,
        }
        if self.superseded_by:
            d["superseded_by"] = self.superseded_by
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Observation":
        return cls(
            id=d["id"],
            entity_id=d["entity_id"],
            content=d["content"],
            source=d.get("source", ""),
            created_at=d.get("created_at", ""),
            deleted=d.get("deleted", False),
            superseded_by=d.get("superseded_by", ""),
        )
