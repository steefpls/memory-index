"""Configuration for memory-index: vault management, paths, model constants."""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DIR = DATA_DIR / "chroma"
LOG_FILE = DATA_DIR / "server.log"
VAULTS_FILE = DATA_DIR / "vaults.json"
ENTITIES_FILE = DATA_DIR / "memory_entities.json"
GRAPH_FILE = DATA_DIR / "memory_graph.json"

# Embedding model — same as code-index (nomic-embed based, handles natural language)
CODERANK_MODEL = "nomic-ai/CodeRankEmbed"
CODERANK_QUERY_PREFIX = "Represent this query for searching relevant knowledge: "
CODERANK_ONNX_DIR = DATA_DIR / "coderank_onnx"


@dataclass
class VaultConfig:
    """Configuration for a single memory vault."""
    name: str
    collection_name: str = ""
    created_at: str = ""

    def __post_init__(self):
        if not self.collection_name:
            self.collection_name = f"memory_{self.name}"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "collection_name": self.collection_name,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "VaultConfig":
        return cls(
            name=d["name"],
            collection_name=d.get("collection_name", f"memory_{d['name']}"),
            created_at=d.get("created_at", ""),
        )


# In-memory vault registry
VAULTS: dict[str, VaultConfig] = {}


def _load_vaults() -> None:
    """Load vault configs from disk."""
    global VAULTS
    if VAULTS_FILE.exists():
        try:
            data = json.loads(VAULTS_FILE.read_text(encoding="utf-8"))
            VAULTS = {
                name: VaultConfig.from_dict(cfg)
                for name, cfg in data.get("vaults", {}).items()
            }
            logger.info("Loaded %d vaults from %s", len(VAULTS), VAULTS_FILE)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load vaults: %s", e)
            VAULTS = {}
    else:
        VAULTS = {}


def _save_vaults() -> None:
    """Save vault configs to disk."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    data = {"vaults": {name: cfg.to_dict() for name, cfg in VAULTS.items()}}
    VAULTS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def create_vault(name: str) -> VaultConfig:
    """Create a new vault. Returns the VaultConfig."""
    from datetime import datetime, timezone
    if name in VAULTS:
        return VAULTS[name]
    vault = VaultConfig(
        name=name,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    VAULTS[name] = vault
    _save_vaults()
    logger.info("Created vault: %s (collection: %s)", name, vault.collection_name)
    return vault


def delete_vault(name: str) -> bool:
    """Delete a vault config. Returns True if deleted."""
    if name not in VAULTS:
        return False
    del VAULTS[name]
    _save_vaults()
    logger.info("Deleted vault: %s", name)
    return True


def get_vault(name: str) -> VaultConfig | None:
    """Get a vault by name."""
    return VAULTS.get(name)


def list_vaults() -> list[VaultConfig]:
    """List all vaults."""
    return list(VAULTS.values())


# Load vaults at import time
_load_vaults()
