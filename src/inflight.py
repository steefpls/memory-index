"""GET /inflight: how many MCP calls this daemon is answering right now.

The hub's deployer asks this right before it restarts the service, and waits
while a call is in flight, so a deploy never cuts a tool call in half.

    GET /inflight  ->  {"in_flight": <int>, "oldest_s": <float or null>}

No key: it says nothing but a count. It is answered here, in an ASGI
middleware wrapped around the whole app, before any route or auth.

What counts: a POST on the MCP path (any key in the path), from the moment it
arrives until its response is fully sent -- a streamed (SSE) answer included,
because the app only returns once the last byte has gone. Not counted: the
long-lived GET stream, DELETE, /inflight itself and every other route.
Counting can never block or fail a request: its own errors are swallowed.

Self-contained on purpose (only the standard library), so each MCP daemon in
the fleet carries its own copy instead of a shared release.
"""
from __future__ import annotations

import itertools
import json
import logging
import time
from typing import Any, Awaitable, Callable

PATH = "/inflight"

Scope = dict[str, Any]
Receive = Callable[[], Awaitable[dict[str, Any]]]
Send = Callable[[dict[str, Any]], Awaitable[None]]


class InflightCounter:
    """Start times of the calls being answered, keyed by a ticket."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._starts: dict[int, float] = {}
        self._tickets = itertools.count(1)

    def begin(self) -> int | None:
        try:
            ticket = next(self._tickets)
            self._starts[ticket] = self._clock()
            return ticket
        except Exception:
            return None

    def end(self, ticket: int | None) -> None:
        try:
            if ticket is not None:
                self._starts.pop(ticket, None)
        except Exception:
            pass

    def snapshot(self) -> dict[str, Any]:
        starts = list(self._starts.values())
        if not starts:
            return {"in_flight": 0, "oldest_s": None}
        return {"in_flight": len(starts),
                "oldest_s": round(max(0.0, self._clock() - min(starts)), 3)}


COUNTER = InflightCounter()


class InflightMiddleware:
    """Counts MCP POSTs and answers GET /inflight; everything else passes."""

    def __init__(self, app: Callable[..., Awaitable[None]], *, mcp_prefix: str = "/mcp",
                 counter: InflightCounter | None = None) -> None:
        self.app = app
        self.mcp_prefix = mcp_prefix.rstrip("/")
        self.counter = counter if counter is not None else COUNTER

    def _is_mcp_call(self, scope: Scope) -> bool:
        try:
            if scope.get("method") != "POST":
                return False
            path = str(scope.get("path") or "").rstrip("/")
            return path == self.mcp_prefix or path.startswith(self.mcp_prefix + "/")
        except Exception:
            return False

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        if scope.get("path") == PATH and scope.get("method") in ("GET", "HEAD"):
            await self._answer(scope, send)
            return
        ticket = self.counter.begin() if self._is_mcp_call(scope) else None
        try:
            await self.app(scope, receive, send)
        finally:
            self.counter.end(ticket)

    async def _answer(self, scope: Scope, send: Send) -> None:
        body = json.dumps(self.counter.snapshot()).encode()
        await send({"type": "http.response.start", "status": 200,
                    "headers": [(b"content-type", b"application/json"),
                                (b"content-length", str(len(body)).encode()),
                                (b"cache-control", b"no-store")]})
        await send({"type": "http.response.body",
                    "body": b"" if scope.get("method") == "HEAD" else body})


class _QuietInflight(logging.Filter):
    """Drops uvicorn's access line for /inflight: the hub polls it."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            args = record.args
            return not (isinstance(args, tuple) and len(args) >= 3
                        and str(args[2]).split("?", 1)[0] == PATH)
        except Exception:
            return True


def quiet_access_log(logger_name: str = "uvicorn.access") -> None:
    """Call after uvicorn.Config is built (it re-configures its loggers)."""
    log = logging.getLogger(logger_name)
    if not any(isinstance(f, _QuietInflight) for f in log.filters):
        log.addFilter(_QuietInflight())


def wrap(mcp: Any, mcp_prefix: str = "/mcp") -> InflightMiddleware:
    """A FastMCP's streamable-HTTP app, counted."""
    return InflightMiddleware(mcp.streamable_http_app(), mcp_prefix=mcp_prefix)


async def serve_streamable_http(mcp: Any, mcp_prefix: str = "/mcp") -> None:
    """FastMCP.run_streamable_http_async, with the /inflight wrapper."""
    import uvicorn

    config = uvicorn.Config(wrap(mcp, mcp_prefix), host=mcp.settings.host,
                            port=mcp.settings.port,
                            log_level=mcp.settings.log_level.lower())
    quiet_access_log()
    await uvicorn.Server(config).serve()
