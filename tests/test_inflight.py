"""GET /inflight counts the MCP calls being answered, for the hub's deployer.

The hub asks /inflight right before it restarts this daemon and waits while a
call is in flight. A slow tool call on the app the daemon serves
(inflight.wrap around FastMCP's streamable-HTTP app) shows in_flight 1 while
it runs and 0 once answered; a streamed answer counts until its last chunk;
the GET stream, DELETE, other routes and /inflight itself never count; a call
that crashes still ends; and uvicorn's access line for /inflight is dropped.

    PYTHONPATH=. python tests/test_inflight.py
"""
import asyncio
import logging
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx  # noqa: E402
from mcp.server.fastmcp import FastMCP  # noqa: E402
from mcp.server.transport_security import TransportSecuritySettings  # noqa: E402
from src import inflight  # noqa: E402

HEADERS = {"accept": "application/json, text/event-stream",
           "content-type": "application/json"}
IDLE = {"in_flight": 0, "oldest_s": None}


def _client(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                             base_url="http://127.0.0.1")


def _a_slow_tool_call_is_in_flight_until_answered():
    async def run():
        inflight.COUNTER = inflight.InflightCounter()
        mcp = FastMCP("t")
        mcp.settings.streamable_http_path = "/mcp/keykeykeykey"
        mcp.settings.json_response = True
        mcp.settings.stateless_http = True
        mcp.settings.transport_security = TransportSecuritySettings(
            enable_dns_rebinding_protection=False)
        started, release = asyncio.Event(), asyncio.Event()

        @mcp.tool()
        async def slow() -> str:
            started.set()
            await release.wait()
            return "done"

        app = inflight.wrap(mcp)       # builds the session manager
        async with mcp.session_manager.run(), _client(app) as client:
            r = await client.get("/inflight")
            assert r.status_code == 200
            assert r.headers["content-type"] == "application/json"
            assert r.json() == IDLE
            call = asyncio.create_task(client.post(
                "/mcp/keykeykeykey", headers=HEADERS,
                json={"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                      "params": {"name": "slow", "arguments": {}}}))
            await asyncio.wait_for(started.wait(), 10)
            snap = (await client.get("/inflight")).json()
            assert snap["in_flight"] == 1
            assert isinstance(snap["oldest_s"], float) and snap["oldest_s"] >= 0
            release.set()
            r = await asyncio.wait_for(call, 10)
            assert r.status_code == 200 and "done" in r.text
            assert (await client.get("/inflight")).json() == IDLE

    asyncio.run(run())


def _streamed_answers_count_to_the_end_and_other_routes_never():
    async def run():
        counter = inflight.InflightCounter()
        gate = asyncio.Event()

        async def app(scope, receive, send):
            if scope["path"] == "/boom":
                raise RuntimeError("boom")
            await send({"type": "http.response.start", "status": 200,
                        "headers": [(b"content-type", b"text/event-stream")]})
            await send({"type": "http.response.body", "body": b"data: 1\n\n", "more_body": True})
            if scope["path"].startswith("/mcp/") and scope["method"] in ("POST", "GET"):
                await gate.wait()
            await send({"type": "http.response.body", "body": b"data: 2\n\n"})

        wrapped = inflight.InflightMiddleware(app, mcp_prefix="/mcp", counter=counter)
        async with _client(wrapped) as client:
            post = asyncio.create_task(client.post("/mcp/anykey", content=b"{}"))
            stream = asyncio.create_task(client.get("/mcp/anykey"))
            await asyncio.sleep(0.05)
            assert counter.snapshot()["in_flight"] == 1     # the POST, not the GET stream
            await client.delete("/mcp/anykey")
            await client.get("/health")
            await client.post("/upload/anykey", content=b"x")
            assert counter.snapshot()["in_flight"] == 1
            gate.set()
            await asyncio.wait_for(asyncio.gather(post, stream), 10)
            assert counter.snapshot() == IDLE
        crash = inflight.InflightMiddleware(app, mcp_prefix="/boom", counter=counter)
        try:
            await crash({"type": "http", "method": "POST", "path": "/boom"}, None, None)
        except RuntimeError:
            pass
        assert counter.snapshot() == IDLE

    asyncio.run(run())


def _the_access_log_leaves_inflight_out():
    inflight.quiet_access_log()
    inflight.quiet_access_log()
    log = logging.getLogger("uvicorn.access")
    quiet = [f for f in log.filters if isinstance(f, inflight._QuietInflight)]
    assert len(quiet) == 1

    def rec(path):
        return logging.LogRecord("uvicorn.access", logging.INFO, __file__, 1,
                                 '%s - "%s %s HTTP/%s" %d',
                                 ("127.0.0.1:1", "GET", path, "1.1", 200), None)
    assert not quiet[0].filter(rec("/inflight"))
    assert quiet[0].filter(rec("/mcp/xyz"))


class InflightTests(unittest.TestCase):
    def test_a_slow_tool_call_is_in_flight_until_answered(self):
        _a_slow_tool_call_is_in_flight_until_answered()

    def test_streamed_answers_count_to_the_end_and_other_routes_never(self):
        _streamed_answers_count_to_the_end_and_other_routes_never()

    def test_the_access_log_leaves_inflight_out(self):
        _the_access_log_leaves_inflight_out()


if __name__ == "__main__":
    unittest.main()
