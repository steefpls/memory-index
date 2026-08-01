# memory-index

Local MCP server for persistent entity/observation/relation memory. Provides semantic search over a knowledge graph via ChromaDB vectors + NetworkX graph layer.

Built for Claude Code / Codex CLI. CPU-only — single-observation embeds are ~10ms, so GPU adds overhead with no benefit.

## Quick Start

```bat
setup.bat
```

This creates a venv, installs deps, exports the ONNX model (~274MB download on first run), and registers the MCP server.

## Tools (26)

### Entities & observations

| Tool | Purpose |
|------|---------|
| `create_entity` | Create a named entity with optional initial observations (`observations` is a list of strings) |
| `get_entity` | Entity details with observations and relations (current and superseded shown separately) |
| `update_entity` | Update an entity's name or type |
| `delete_entity` | Soft delete an entity and its observations |
| `list_entities` | Paginated list, filter by vault / type |
| `add_observation` | Add one observation to an entity (supports `supersedes` and `occurred_at`) |
| `add_observations` | Add multiple observations to one entity in a single call — one embed pass, one write |
| `delete_observation` | Remove an observation by ID |

### Relations & graph

| Tool | Purpose |
|------|---------|
| `create_relation` | Create a directed relation between two entities |
| `delete_relation` | Remove a relation by ID |
| `get_neighbors` | Entities connected to a given entity, up to `max_depth` hops |
| `analyze_graph` | PageRank centrality, Louvain community detection, under-documented entity (knowledge gap) identification |
| `run_librarian` | Discover knowledge clusters and structural gaps |
| `visualize_graph` | Generate an interactive graph visualization and open it in a browser |
| `get_graph_summary` | Node/edge counts, connected components, relation type distribution |

### Search & time

| Tool | Purpose |
|------|---------|
| `search_memory` | Observation-level semantic search with calibrated confidence, temporal / type / superseded filters, opt-in graph expansion |
| `query_timeline` | Observations across a time range, ordered chronologically |
| `point_in_time` | What was known about an entity as of a given timestamp |
| `get_temporal_neighbors` | Entities whose observations sit temporally adjacent to a given entity's |

### Vaults, status & portability

| Tool | Purpose |
|------|---------|
| `memory_status` | Health check: backend, entity/observation/relation counts per vault |
| `list_vaults` | List vaults with entity and observation counts |
| `create_vault` | Create a new vault for isolating a knowledge domain |
| `delete_vault` | Delete a vault and all its entities, observations, relations, and vectors |
| `export_vault` | Export a vault to a portable zip archive |
| `import_vault` | Import a vault export zip into a target vault (additive, lossless — supersede history is preserved) |
| `vacuum_store` | Hard-remove stale rows left behind by soft-delete semantics |

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
- `export_vault` / `import_vault` round-trip superseded rows and remap their pointers, so history survives a migration

### Event time vs write time

Every observation records `created_at` (when it was written). An observation may also carry `occurred_at` — when the fact was actually true — set explicitly at write time. Temporal tools order and window on the *effective* timestamp (`occurred_at` when present, else `created_at`), so backfilled history sorts where it belongs rather than clustering at the import date.

`scripts/backfill_occurred_at.py` infers `occurred_at` for existing observations that contain exactly one unambiguous ISO date in their text. It is dry-run by default; pass `--apply` to write.

### Temporal Queries

Search supports `since` and `before` filters using ISO date/datetime:

```
search_memory("framework migration", since="2026-03-01", before="2026-03-14")
```

## Agent Usage

Rules for LLM agents (Claude Code, Codex CLI, etc.) writing to and reading from memory-index. Memory-index does not police these rules — they must be enforced at the agent layer via your `CLAUDE.md` or system prompt.

### CLAUDE.md Snippet

Copy this block verbatim into your `CLAUDE.md` (or `agents.md` / system prompt). Adjust the vault name to match your setup.

````markdown
## MANDATORY: Memory via memory-index MCP

**Use `memory-index` MCP tools as the primary memory system. Default vault: `work`.**

### READ — search before answering (non-negotiable)
Before answering the user's first message in any conversation, call `search_memory` (vault: `work`). Derive the query from what the user is asking about — people, projects, decisions, codebases, anything plausibly stored.
- Always search before claiming you don't know something.
- Re-search when new topics surface mid-conversation, or before making architectural/process suggestions.
- If `search_memory` isn't loaded yet, fetch it with `ToolSearch` first, then search.

### WRITE — search before writing (non-negotiable)
Before `create_entity`, `add_observation`, or `create_relation`, call `search_memory` first. Then choose in order:
1. `add_observation` on an existing entity (most common)
2. `create_relation` between existing entities (fact is implicit in the edge)
3. `create_entity` only if nothing relevant exists

### Atomicity — one observation = one atomic fact
Each observation is embedded individually. Split multi-fact statements into separate `add_observation` calls — one fact per call, always. No JSON arrays, comma-packed lists, or summary bundles. Cleanup is via per-observation supersession (`supersedes=<old_id>`), not by overwriting.

### Attribution — facts about X live on X
Ask: what is the *subject* of this fact? That entity is where the observation lives. Use relations to connect across entities rather than duplicating facts on both sides.

### Source + date discipline
Always populate the `source` field — chat date, filename, PR number, meeting context. Convert relative dates ("yesterday", "last Thursday") to absolute ISO dates in observation content.

### Entity types
`person`, `project`, `concept`, `decision`, `error`, `solution`, `technology`, `pattern`, `preference`, `organization`, `event`, `reference`

### What NOT to save
- Ephemeral state ("currently debugging X")
- Facts derivable from code, git, or files
- Anything already in CLAUDE.md
- Conversation transcripts — record the *outcome*, not the discussion

### Before recommending from memory
A memory naming a file, function, or flag is a claim it existed when written. Verify before acting: check the file exists, grep for the symbol. If a recalled memory conflicts with what you observe now, trust what you observe and update the stale memory.
````

### Read before answering

Call `search_memory` before responding to questions that could plausibly involve stored context — people, projects, decisions, preferences, prior work. Never claim "I don't know" about the user's world without searching first. Re-search mid-conversation when new topics surface or before making recommendations that depend on prior decisions.

### Search before writing

Before `create_entity`, `add_observation`, or `create_relation`, call `search_memory` first. Then choose, in order of preference:

1. `add_observation` on an existing entity (most common — entity exists, fact is new)
2. `create_relation` between existing entities (the fact is implicit in the edge)
3. `create_entity` only if nothing relevant exists

Never duplicate facts. If fact X already lives on entity Y, link via relation — don't restate.

### Atomicity — one observation = one atomic fact

Observations are embedded individually. A packed observation ("Likes X, works at Y, based in Z") produces one embedding that represents none of those facts well, and semantic search degrades at the fact level.

- One fact per `add_observation` call. Five facts → five calls (or one `add_observations` call with five list items — same granularity, one round trip).
- No JSON arrays, comma-packed lists, or "consolidated" mega-observations.
- Self-contained facts may rely on parent entity context — on entity `Alice`, the observation "Based in Singapore" is fine.
- Cleanup happens via per-observation supersession (`supersedes=<old_id>`), never via bundling.

### Attribution — facts about X live on X

Ask: what is the *subject* of this fact? That's where the observation lives.

- "Project P forks library L" → on entity `P`.
- "Alice built P in one afternoon" → on entity `Alice` (the fact is about Alice's velocity).
- "Company C laid off the QA team" → on entity `C`.
- "Alice's salary at C is $X" → on entity `Alice`.

Use relations liberally to connect across entities (`Alice -[works_at]-> C`, `Alice -[created]-> P`) rather than duplicating facts on both sides.

### Source + date discipline

- Populate the `source` field on every observation — chat date, file name, conversation context, web research summary. Sources make observations traceable and trustworthy later.
- Convert relative dates to absolute in observation content. "Yesterday" → actual ISO date. "Last Thursday" → actual ISO date.

### When NOT to write

Skip writing observations that are:

- Ephemeral conversation state ("currently debugging X", "in the middle of Y")
- Derivable from code, git, or files (paths, function names, recent commits)
- Already captured in `CLAUDE.md` or equivalent instruction files
- Conversation transcripts or "we discussed X today" notes — record the *outcome*, not the discussion
- Stale project snapshots ("Project P has N modules as of March") — these rot fast; `git log` is authoritative

### Memory is testimony, not ground truth

A stored observation is a claim someone made at a time, not a verified fact about the present. An observation naming a file, function, config flag, or version number asserts that it existed when the observation was written — nothing more. Before acting on such a memory, verify it: check the path exists, grep for the symbol, read the current config. When a recalled memory conflicts with what you can observe right now, **what you observe wins**, and the stale memory should be superseded rather than left to mislead the next session.

The confidence labels search returns are calibrated distances, not truth values. `HIGH` means "this text is close to your query in embedding space" — it says nothing about whether the claim is still correct.

### Entity type conventions

Suggested types (free-form strings — no enum enforced):

`person`, `project`, `concept`, `decision`, `error`, `solution`, `technology`, `pattern`, `preference`, `organization`, `event`, `reference`

### Default vault

Unless the agent is explicitly told otherwise, all operations should target a single named vault (e.g. `work`). Multi-vault setups are for domain isolation (personal vs work vs a specific client), not for sharding a single domain.

## Search

**The unit of retrieval is the observation, not the entity.** A result is one fact, rendered with its entity as context — not an entity with a sample of its facts attached. Two facts from the same entity are two results and compete on their own merit; a fact from a sparsely-connected entity is not penalized for its neighbourhood.

Pipeline:

1. Embed the query with EmbeddingGemma-300m (CPU, ONNX q8), using the retrieval-query prefix
2. Query each in-scope vault's ChromaDB collection, applying `entity_type`, `since`, `before`, and superseded filters in the vector store's `where` clause
3. Flatten every matching observation into one list across vaults and rank it by calibrated relevance score (a strictly decreasing function of distance within a vault, and the only key that compares fairly across vaults when searching all of them)
4. Gate on the vault's calibrated noise floor — an observation is *above threshold* when its band is anything other than `NO MATCH`
5. Select results (below)
6. Optionally expand via the graph (below)
7. Render as ranked facts grouped under one line of entity context (`text`), or a flat ranked list of observation objects including observation IDs (`json`)

### Top-5 / min-3 selection rule

- If **3 or more** observations clear the threshold, return the top `n_results` of them (default 5). An explicitly smaller `n_results` is honoured.
- If **fewer than 3** clear it, return the best 3 overall anyway, with their real `LOW` / `NO MATCH` labels intact and a note that the fallback fired.

The floor exists so a search never returns a bare "no results" when the vault does hold something adjacent — the caller sees the near-misses and judges them, rather than being told nothing exists. It is a floor for the threshold-shortfall case, not an override of a caller who asked for fewer results.

### Strategies

- **`semantic`** (default) — vector search only. No graph traversal happens at all.
- **`associative`** (opt-in) — spreading activation over the relation graph *nominates* candidate entities from the semantic hits' neighbourhood. Nomination is not a score: each nominated entity's observations are then queried with the real query embedding and admitted only if they clear the same threshold as a direct hit. Nothing invents a distance, so a well-connected entity cannot outrank a genuine match, and when no neighbour earns its place `associative` degrades silently to `semantic`.

Reciprocal Rank Fusion is no longer used for merging. RRF is rank-only and therefore has no notion of match quality — a popularity-ranked graph neighbour could outrank a real semantic hit. Scoring graph candidates against the actual query replaced it.

### Calibration

Confidence thresholds (`HIGH` / `MEDIUM` / `LOW`, with everything beyond `LOW` reading as `NO MATCH`) are derived per vault: real observations sampled from the vault act as "should match" probes and gibberish strings as "should not match" probes, and the bands are fitted to the gap between the two distributions. This keeps scores meaningful regardless of the vault's content domain.

- Probes are sampled **uniformly at random across the whole collection**, so thresholds track current content instead of anchoring to whatever was written first
- Auto-recalibrates every 10 observations per vault (~480ms), off the write lock
- Falls back to generic probes if the vault holds fewer than 10 observations
- `scripts/recalibrate.py` forces a recalibration pass

## Storage & concurrency

- Entity/observation JSON and the graph JSON are written **atomically** — write to a sibling temp file, `fsync`, then `os.replace` — so a crash mid-write can never truncate the store.
- All store mutations and reads are guarded by a process-wide `RLock`, so concurrent MCP calls cannot interleave into a lost update. Expensive work (embedding, calibration, clustering) runs *outside* the lock so it never stalls readers.
- ChromaDB I/O sits outside the lock; only metadata preparation is inside it.

## Architecture

Forked from [code-index](https://github.com/you/code-index). Same embedding pipeline shape, simplified to CPU-only; the model was swapped from CodeRankEmbed (a code retriever) to google/embeddinggemma-300m (768-dim, ONNX q8) for natural-language memory retrieval.

```
src/
├── server.py              # FastMCP, 26 tool registrations
├── config.py              # VaultConfig, vault CRUD, paths
├── indexer/
│   ├── embedder.py        # ONNX CPU embedder singleton + ChromaDB client
│   ├── calibration.py     # Per-vault distance thresholds (randomly sampled probes)
│   └── store.py           # Entity/observation CRUD, atomic writes, RLock, auto-recalibrate
├── graph/
│   ├── manager.py         # NetworkX MultiDiGraph, atomic JSON persistence
│   └── traversal.py       # Neighbors, spreading activation (nomination only)
├── models/
│   ├── entity.py
│   ├── observation.py     # content, source, created_at, occurred_at, superseded_by
│   └── relation.py
└── tools/
    ├── search.py          # Observation-level ranked search + threshold gate
    ├── entities.py        # Entity/observation tool impls
    ├── relations.py       # Relation tool impls
    ├── temporal.py        # Timeline, point-in-time, temporal neighbors
    ├── graph_analysis.py  # PageRank, Louvain communities, knowledge gaps
    ├── librarian.py       # Cluster + gap discovery
    ├── portability.py     # Vault export/import (lossless supersede history)
    ├── maintenance.py     # vacuum_store
    ├── visualize.py       # Interactive HTML graph
    └── status.py          # Health check, vault management

scripts/
├── download_model.py         # Fetch/export the EmbeddingGemma ONNX model
├── recalibrate.py            # Force per-vault threshold recalibration
├── reembed_all.py            # Re-embed every observation (model change)
├── backfill_occurred_at.py   # Infer occurred_at from ISO dates in text (dry-run default)
└── backup_to_drive.py        # Vault backup
```

## Tests

No pytest — the suites are plain `unittest` scripts:

```bash
PYTHONPATH=. python tests/test_search.py
```

Run them all:

```bash
for f in tests/test_*.py; do PYTHONPATH=. python "$f"; done
```

194 tests across 9 files, covering entity CRUD and superseding, batched observation writes, atomic-write and concurrency behaviour, observation-level search ranking and threshold selection, graph traversal and analysis, the librarian, temporal queries with event time, vault export/import round-trips, maintenance, and the tool layer. The embedder is mocked throughout, so the suite runs without the ONNX model present.
