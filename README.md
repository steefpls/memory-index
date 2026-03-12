# memory-index

Local MCP server for persistent entity/observation/relation memory. Provides semantic search over a knowledge graph via ChromaDB vectors + NetworkX graph layer.

Built for Claude Code / Codex CLI. CPU-only — single-observation embeds are ~10ms, so GPU adds overhead with no benefit.

## Quick Start

```bat
setup.bat
```

This creates a venv, installs deps, exports the ONNX model (~274MB download on first run), and registers the MCP server.

## Tools (16)

| Tool | Purpose |
|------|---------|
| `create_entity` | Create entity + optional initial observations |
| `get_entity` | Entity + observations + relations (shows superseded history) |
| `update_entity` | Update name/type |
| `delete_entity` | Soft delete |
| `list_entities` | Paginated list, filter by vault/type |
| `add_observation` | Add + embed observation to entity (supports `supersedes` to replace old facts) |
| `delete_observation` | Remove observation |
| `create_relation` | Add directed edge in graph |
| `delete_relation` | Remove edge |
| `search_memory` | Hybrid vector + graph-boosted search with temporal and superseded filters |
| `get_neighbors` | Graph traversal from entity |
| `get_graph_summary` | Stats: counts, components, type distribution |
| `memory_status` | Health check (backend, counts) |
| `list_vaults` | Show vaults |
| `create_vault` | Add new vault |
| `delete_vault` | Remove vault + all data |

## Data Model

- **Entities** — named nodes: person, project, concept, decision, technology, etc.
- **Observations** — facts attached to entities, embedded in ChromaDB for semantic retrieval
- **Relations** — directed edges (uses, depends_on, solves, part_of, etc.) stored in NetworkX
- **Vaults** — isolated memory collections, one ChromaDB collection each

### Observation Superseding

When facts change over time, new observations can replace old ones:

```
add_observation("Perception", "Migrated to .NET 8", supersedes="<old_obs_id>")
```

- Old observation is tagged as superseded, not deleted
- Default search excludes superseded observations (returns only current facts)
- Use `include_superseded=True` to search historical data
- `get_entity` shows both current and superseded observations separately
- Supports supersede chains (v1 -> v2 -> v3)

### Temporal Queries

Search supports `since` and `before` filters using ISO date/datetime:

```
search_memory("framework migration", since="2026-03-01", before="2026-03-14")
```

## Search

1. Embed query with CodeRankEmbed (CPU, ONNX)
2. ChromaDB similarity search across vault(s), with optional temporal + entity type filters
3. Filter superseded observations (unless `include_superseded=True`)
4. Deduplicate by entity, merge observations
5. Graph-boost: expand 1-hop neighbors with lower weight
6. Score normalization via per-vault calibration thresholds
7. Return entities + matched observations + confidence

### Calibration

Confidence thresholds are derived per-vault by sampling real observations as "should match" probes and gibberish as "should not match" probes. This makes scores meaningful regardless of vault content domain.

- Auto-recalibrates every 10 observations per vault (~480ms)
- Uses second-nearest neighbor distances (sampled observations match themselves at ~0)
- Falls back to generic probes if vault has < 10 observations

## Architecture

Forked from [code-index](https://github.com/you/code-index). Same embedding pipeline (CodeRankEmbed, 137M params, 768-dim), simplified to CPU-only.

```
src/
├── server.py              # FastMCP, 16 tool registrations
├── config.py              # VaultConfig, vault CRUD, paths
├── hardware.py            # CPU thread detection
├── indexer/
│   ├── embedder.py        # ONNX CPU embedder + ChromaDB client
│   ├── calibration.py     # Per-vault distance thresholds (sampled probes)
│   └── store.py           # Entity/observation CRUD + ChromaDB storage + auto-recalibrate
├── graph/
│   ├── manager.py         # NetworkX MultiDiGraph, JSON persistence
│   └── traversal.py       # Neighbors, graph-boost scoring
├── models/
│   ├── entity.py, observation.py, relation.py
└── tools/
    ├── search.py           # Hybrid vector + graph search + temporal + superseded filters
    ├── entities.py         # Entity/observation tool impls
    ├── relations.py        # Relation tool impls
    └── status.py           # Health check, vault management
```

## Tests

```bash
PYTHONPATH=. python -m pytest tests/ -v
```

54 tests covering entity CRUD, observation superseding, graph traversal, scoring/calibration, and tool layer.
