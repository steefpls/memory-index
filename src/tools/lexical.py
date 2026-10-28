"""BM25 over observations, the second candidate source for search.

The vector pass misses facts whose words match the query but whose embedding
sits elsewhere, which is most of what agents type: keyword lists like "folder
ingestion Sonnet batch approval nudge". Measured 2026-10-28 on the 86 labelled
queries (scratch/memeval/hybrid_eval.py): 27 of 303 must-have facts were
outside the vector top 30, and all 27 were in BM25's top 15.

The index covers every live observation (entity name + content) and is
rebuilt only when the store changes. Callers filter by vault, type, date and
supersession themselves; this module only ranks.
"""

import collections
import math
import re
import threading

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "the a an of to and or for in on at is are was with by from about what who "
    "how when which does do steve steven s his her".split())
_K1 = 1.2
_B = 0.75


def tokens(text: str) -> list[str]:
    return [w for w in _TOKEN.findall((text or "").lower()) if w not in _STOP]


class _Index:
    def __init__(self, docs: dict[str, str]):
        self.postings: dict[str, list[tuple[str, int]]] = collections.defaultdict(list)
        self.length: dict[str, int] = {}
        for oid, text in docs.items():
            toks = tokens(text)
            self.length[oid] = len(toks)
            for word, tf in collections.Counter(toks).items():
                self.postings[word].append((oid, tf))
        self.n = len(docs)
        self.avg = (sum(self.length.values()) / self.n) if self.n else 1.0

    def rank(self, query: str, allowed, limit: int) -> list[str]:
        scores: dict[str, float] = collections.defaultdict(float)
        for word in set(tokens(query)):
            posting = self.postings.get(word)
            if not posting:
                continue
            idf = math.log(1 + (self.n - len(posting) + 0.5) / (len(posting) + 0.5))
            for oid, tf in posting:
                norm = _K1 * (1 - _B + _B * self.length[oid] / self.avg)
                scores[oid] += idf * tf * (_K1 + 1) / (tf + norm)
        out = []
        for oid in sorted(scores, key=lambda oid: -scores[oid]):
            if len(out) == limit:
                break
            if allowed(oid):
                out.append(oid)
        return out


_lock = threading.Lock()
_cache: dict = {"key": None, "index": None}


def search(query: str, allowed, limit: int) -> list[str]:
    """Observation ids best-first by BM25, keeping only those `allowed(oid)`
    accepts, at most `limit`. [] when no query word appears anywhere."""
    from src.indexer.store import snapshot_store
    entities, observations = snapshot_store()
    live = []
    for obs in observations.values():
        ent = entities.get(obs.entity_id)
        if obs.deleted or ent is None or ent.deleted:
            continue
        live.append((obs.id, obs.content or "", ent.name or ""))
    # Python caches each string's hash, so this is a pass over ids, not over
    # text; any write, delete or rename changes it.
    key = (len(live), sum(hash(row) for row in live))
    with _lock:
        if _cache["key"] != key:
            _cache["index"] = _Index({oid: f"{name} {content}" for oid, content, name in live})
            _cache["key"] = key
        index = _cache["index"]
    return index.rank(query, allowed, limit)
