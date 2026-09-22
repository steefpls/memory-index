"""POST /embed — vectors from the already-loaded embedding model.

Other fleet services (orchestrator-hub's run search, first) need the same
embeddings memory-index computes, and loading a second EmbeddingGemma
copy in their own process would double the RAM and CPU the model costs.
This route lends them the singleton this server already holds.

Request:  {"texts": ["...", ...], "kind": "document" | "query"}
Response: {"model": ..., "backend": ..., "dim": 768, "kind": ...,
           "vectors": [[...], ...]}

`kind` picks the prompt prefix the model was trained with: "document"
(default) is how memory-index stores observations, "query" is how it
embeds a search. Vectors are L2-normalised (the ONNX graph does it), so a
dot product is the cosine.

Auth is the MCP key in ``Authorization: Bearer <key>``; a wrong key gets
401. Keeping it out of the URL prevents access logs from recording it.
Embedding runs in a worker thread, one batch at a time, so a caller
backfilling thousands of texts never blocks the MCP event loop or stacks
up model runs next to a live search.
"""
from __future__ import annotations

import asyncio
import hmac
import logging
from typing import Any, Callable

import anyio
from starlette.requests import Request
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)

MODEL_NAME = "embeddinggemma-300m"
MAX_TEXTS = 64
MAX_CHARS = 8000
MAX_BODY_BYTES = 2_000_000
KINDS = ("document", "query")


def _default_embedder() -> Any:
    from src.tools.search import _ensure_backend
    return _ensure_backend()


def _default_backend() -> str:
    from src.indexer.embedder import get_active_backend
    return get_active_backend()


def _default_device() -> dict:
    from src.indexer.embedder import get_embed_device
    return get_embed_device()


def make_health_endpoint(
    get_backend: Callable[[], str] = _default_backend,
    get_device: Callable[[], dict] = _default_device,
):
    """GET /health: unauthenticated, no secrets, what the hub's watchdog reads.

    `embed_device` is {"requested", "active", "error"}: `active` is None
    until the first embed loads the model, and `error` names why a wanted
    CUDA session is not the one running. Nothing here is keyed because it
    says nothing about the vault, only about the process."""
    async def health(request: Request) -> JSONResponse:
        return JSONResponse({
            "ok": True,
            "model": MODEL_NAME,
            "backend": get_backend(),
            "embed_device": get_device(),
        })
    return health


def make_embed_endpoint(
    api_key: str,
    get_embedder: Callable[[], Any] = _default_embedder,
    get_backend: Callable[[], str] = _default_backend,
):
    """The route handler, closed over the key and the model accessors (the
    tests pass fakes)."""
    lock = asyncio.Lock()
    expected = api_key.encode("utf-8")

    async def embed(request: Request) -> JSONResponse:
        auth = request.headers.get("authorization", "")
        given = auth[7:].strip().encode("utf-8") if auth.lower().startswith("bearer ") else b""
        if not expected or not hmac.compare_digest(given, expected):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        body = await request.body()
        if len(body) > MAX_BODY_BYTES:
            return JSONResponse({"error": f"body over {MAX_BODY_BYTES} bytes"}, status_code=413)
        try:
            payload = await request.json() if body else {}
        except ValueError:
            return JSONResponse({"error": "body must be JSON"}, status_code=400)
        if not isinstance(payload, dict):
            return JSONResponse({"error": "body must be a JSON object"}, status_code=400)
        texts = payload.get("texts")
        kind = payload.get("kind") or "document"
        if kind not in KINDS:
            return JSONResponse({"error": f"kind must be one of {list(KINDS)}"}, status_code=400)
        if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
            return JSONResponse({"error": "texts must be a list of strings"}, status_code=400)
        if len(texts) > MAX_TEXTS:
            return JSONResponse({"error": f"at most {MAX_TEXTS} texts per call"}, status_code=413)
        if not texts:
            return JSONResponse({"model": MODEL_NAME, "backend": get_backend(),
                                 "dim": 0, "kind": kind, "vectors": []})
        # An empty string embeds to the bare prefix; a single space keeps
        # the batch shape without inventing content.
        clipped = [(t[:MAX_CHARS] if t.strip() else " ") for t in texts]

        def run() -> list[list[float]]:
            fn = get_embedder()
            if kind == "query":
                return fn.embed_queries(clipped)
            return fn(clipped)

        try:
            async with lock:
                vectors = await anyio.to_thread.run_sync(run)
        except TimeoutError as e:
            return JSONResponse({"error": f"model not ready: {e}"}, status_code=503)
        except Exception as e:  # noqa: BLE001 — report, don't crash the daemon
            logger.exception("embed endpoint failed")
            return JSONResponse({"error": f"embedding failed: {type(e).__name__}"}, status_code=500)
        out = [[round(float(x), 6) for x in v] for v in vectors]
        return JSONResponse({
            "model": MODEL_NAME,
            "backend": get_backend(),
            "dim": len(out[0]) if out else 0,
            "kind": kind,
            "vectors": out,
        })

    return embed


def register(mcp: Any, api_key: str) -> None:
    """Add the Bearer-authenticated POST /embed route and the open GET /health."""
    mcp.custom_route("/embed", methods=["POST"], name="embed")(make_embed_endpoint(api_key))
    mcp.custom_route("/health", methods=["GET"], name="health")(make_health_endpoint())
