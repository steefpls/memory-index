"""Tests for POST /embed with Bearer auth (src/embed_http.py).

The real model is never loaded: the endpoint takes its embedder accessor as
an argument, and these tests hand it a fake that records what it was asked.
"""

import os
import sys
import threading
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from starlette.applications import Starlette  # noqa: E402
from starlette.routing import Route  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402

from src.embed_http import MAX_TEXTS, make_embed_endpoint, make_health_endpoint  # noqa: E402

KEY = "fakekey0123456789abcdef"


class FakeEmbedder:
    def __init__(self):
        self.calls = []
        self.threads = set()

    def _vec(self, t):
        return [float(len(t)), 0.5, -0.25]

    def __call__(self, texts):
        self.calls.append(("document", list(texts)))
        self.threads.add(threading.get_ident())
        return [self._vec(t) for t in texts]

    def embed_queries(self, texts):
        self.calls.append(("query", list(texts)))
        return [self._vec(t) for t in texts]


def _client(fake, key=KEY):
    handler = make_embed_endpoint(key, get_embedder=lambda: fake, get_backend=lambda: "fake")
    app = Starlette(routes=[Route("/embed", handler, methods=["POST"])])
    return TestClient(app)


def _post(client, *, key=KEY, **kwargs):
    headers = dict(kwargs.pop("headers", {}))
    headers["Authorization"] = f"Bearer {key}"
    return client.post("/embed", headers=headers, **kwargs)


class TestEmbedEndpoint(unittest.TestCase):
    def test_wrong_key_is_401(self):
        fake = FakeEmbedder()
        r = _post(_client(fake), key="wrongkey", json={"texts": ["a"]})
        self.assertEqual(r.status_code, 401)
        self.assertEqual(r.json(), {"error": "unauthorized"})
        self.assertEqual(fake.calls, [])

    def test_empty_configured_key_never_authorises(self):
        r = _post(_client(FakeEmbedder(), key=""), key="x", json={"texts": ["a"]})
        self.assertEqual(r.status_code, 401)

    def test_missing_or_non_bearer_auth_is_401(self):
        c = _client(FakeEmbedder())
        self.assertEqual(c.post("/embed", json={"texts": ["a"]}).status_code, 401)
        self.assertEqual(c.post("/embed", headers={"Authorization": KEY}, json={"texts": ["a"]}).status_code, 401)

    def test_documents_by_default(self):
        fake = FakeEmbedder()
        r = _post(_client(fake), json={"texts": ["hello", "hi"]})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["kind"], "document")
        self.assertEqual(body["dim"], 3)
        self.assertEqual(body["backend"], "fake")
        self.assertEqual(body["vectors"], [[5.0, 0.5, -0.25], [2.0, 0.5, -0.25]])
        self.assertEqual(fake.calls, [("document", ["hello", "hi"])])
        # Ran off the event loop's thread.
        self.assertNotIn(threading.get_ident(), fake.threads)

    def test_query_kind_uses_query_prefix_path(self):
        fake = FakeEmbedder()
        r = _post(_client(fake), json={"texts": ["what did I ask"], "kind": "query"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(fake.calls, [("query", ["what did I ask"])])

    def test_validation(self):
        c = _client(FakeEmbedder())
        self.assertEqual(_post(c, json={"texts": "nope"}).status_code, 400)
        self.assertEqual(_post(c, json={"texts": [1]}).status_code, 400)
        self.assertEqual(_post(c, json={"texts": ["a"], "kind": "x"}).status_code, 400)
        self.assertEqual(_post(c, content=b"not json").status_code, 400)
        too_many = {"texts": ["a"] * (MAX_TEXTS + 1)}
        self.assertEqual(_post(c, json=too_many).status_code, 413)

    def test_empty_list_and_blank_text(self):
        fake = FakeEmbedder()
        c = _client(fake)
        r = _post(c, json={"texts": []})
        self.assertEqual(r.json()["vectors"], [])
        self.assertEqual(fake.calls, [])
        r = _post(c, json={"texts": ["", "x" * 10000]})
        self.assertEqual(r.status_code, 200)
        sent = fake.calls[-1][1]
        self.assertEqual(sent[0], " ")
        self.assertEqual(len(sent[1]), 8000)

    def test_model_failure_is_reported_not_raised(self):
        class Broken(FakeEmbedder):
            def __call__(self, texts):
                raise RuntimeError("onnx blew up")

        r = _post(_client(Broken()), json={"texts": ["a"]})
        self.assertEqual(r.status_code, 500)
        self.assertIn("RuntimeError", r.json()["error"])

    def test_model_not_ready_is_503(self):
        def not_ready():
            raise TimeoutError("init timed out")

        handler = make_embed_endpoint(KEY, get_embedder=not_ready, get_backend=lambda: "not initialized")
        app = Starlette(routes=[Route("/embed", handler, methods=["POST"])])
        r = _post(TestClient(app), json={"texts": ["a"]})
        self.assertEqual(r.status_code, 503)

    def test_registers_on_fastmcp(self):
        from mcp.server.fastmcp import FastMCP
        from src.embed_http import register

        m = FastMCP("t")
        register(m, KEY)
        paths = [getattr(r, "path", None) for r in m._custom_starlette_routes]
        self.assertIn("/embed", paths)

    def test_health_is_open_and_reports_the_device(self):
        handler = make_health_endpoint(
            get_backend=lambda: "ONNX + CUDA",
            get_device=lambda: {"requested": "auto", "active": "cuda", "error": None},
            get_load_error=lambda: None)
        app = Starlette(routes=[Route("/health", handler, methods=["GET"])])
        r = TestClient(app).get("/health")  # no Authorization header
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertTrue(body["ok"])
        self.assertEqual(body["backend"], "ONNX + CUDA")
        self.assertEqual(body["embed_device"], {"requested": "auto", "active": "cuda", "error": None})

    def test_health_fails_when_the_model_wont_load(self):
        handler = make_health_endpoint(
            get_backend=lambda: "not initialized",
            get_device=lambda: {"requested": "cuda", "active": None, "error": None},
            get_load_error=lambda: "AttributeError: module 'onnxruntime' has no attribute 'SessionOptions'")
        app = Starlette(routes=[Route("/health", handler, methods=["GET"])])
        r = TestClient(app).get("/health")
        self.assertEqual(r.status_code, 503)
        body = r.json()
        self.assertFalse(body["ok"])
        self.assertIn("SessionOptions", body["error"])

    def test_health_registers_beside_embed(self):
        from mcp.server.fastmcp import FastMCP
        from src.embed_http import register

        m = FastMCP("t")
        register(m, KEY)
        paths = [getattr(r, "path", None) for r in m._custom_starlette_routes]
        self.assertIn("/health", paths)


if __name__ == "__main__":
    unittest.main()
