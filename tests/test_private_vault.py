"""The private vault: only the box's owner may use it (src/access.py).

Three layers:
  - who a call is for (src/requester.py), for every caller class: the owner's
    chat turns and sessions, the other owner, the peer assistant, a trusted
    non-owner, their wakes and jobs, programme steps, schedules, the gardener,
    the hub's tool gateway, a chat daemon calling for someone, and the ways a
    lookup can fail (each refused);
  - every MCP tool, called through server.py as a non-owner and as the owner;
  - the HTTP daemon itself: a request's caller reaches its tool call, and a
    call that has lost its request is refused.
"""

import asyncio
import json
import os
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import access, requester  # noqa: E402
from src.access import Verdict  # noqa: E402

OWNER = Verdict(True, "test: the owner")
NOT_OWNER = Verdict(False, "test: someone else")


# ---------------------------------------------------------------------------
# 1. Who a call is for
# ---------------------------------------------------------------------------

class FakeProc:
    def __init__(self, pid, name, env=None, parent=None, created=100.0):
        self.pid, self._name, self._env, self._parent, self._created = pid, name, env or {}, parent, created

    def name(self):
        return self._name

    def environ(self):
        return dict(self._env)

    def parent(self):
        return self._parent

    def create_time(self):
        return self._created


class FakeProcs:
    """A process table: {pid: FakeProc}, which pid holds which client port,
    and which pids listen on a fleet port."""

    def __init__(self, procs, conns, listeners):
        self.procs, self.conns, self.listeners = procs, conns, listeners

    def pid_at(self, ip, port):
        return self.conns.get(port)

    def services(self, ports):
        return {pid: port for pid, port in self.listeners.items() if port in ports}

    def proc(self, pid):
        if pid not in self.procs:
            raise LookupError(pid)
        return self.procs[pid]


def chain(*links, port=50000, listeners=None):
    """FakeProcs for one chain, the first link holding the connection and
    each next link its parent. A link is (name, env)."""
    procs, parent = {}, None
    pid = 1000 + len(links)
    for name, env in reversed(links):
        p = FakeProc(pid, name, env, parent)
        procs[pid] = p
        parent = p
        pid -= 1
    first = parent.pid
    named = {p._name: p.pid for p in procs.values()}
    ports = {named[n]: prt for n, prt in (listeners or {}).items()}
    return FakeProcs(procs, {port: first}, ports)


class FakeHub:
    def __init__(self, triggers=None, routines=None, programmes=None, down=False):
        self.triggers = triggers or {}
        self.routines = routines or {}
        self.programmes = programmes or {}
        self.down = down
        self.asked = []

    def trigger(self, tid):
        self.asked.append(tid)
        if self.down:
            raise requester.HubError("hub unreachable")
        return self.triggers.get(tid)

    def routine(self, rid):
        if self.down:
            raise requester.HubError("hub unreachable")
        return self.routines.get(rid)

    def programme(self, pid):
        if self.down:
            raise requester.HubError("hub unreachable")
        return self.programmes.get(pid)


def chat_row(source, owner, name, **extra):
    return {"source": source, "sender_owner": owner, "sender_name": name,
            "sender_id": "1", "execution": {}, **extra}


def hub_row(source="hub", **extra):
    ex = extra.pop("execution", {})
    return {"source": source, "sender_owner": None, "execution": ex, **extra}


HUB = FakeHub(
    triggers={
        "steve-tg": chat_row("tg", True, "Steve"),
        "steve-dc": chat_row("dcbot", True, "Steve"),
        "sk-wa": chat_row("wa", False, "See Kiat (SK) KOH"),
        "peer-tg": chat_row("tg", False, "Friday (peer)"),
        "trusted-tg": chat_row("tg", False, "LOuie"),
        "unknown-bot": chat_row("bot", False, "Unknown"),
        "no-sender": {"source": "tg", "execution": {}},
        # A wake runs as the sender whose turn fired the job.
        "wake-for-sk": chat_row("wa", False, "See Kiat (SK) KOH"),
        "wake-for-steve": chat_row("tg", True, "Steve"),
        # Jobs fired from chat turns (run_fire with X-Hub-Chat).
        "job-from-sk": hub_row(parent_id="sk-wa", execution={"parent_id": "sk-wa", "requested_by": "agent"}),
        "job-from-steve": hub_row(parent_id="steve-dc"),
        "subjob-from-sk": hub_row(parent_id="job-from-sk"),
        "retry-of-sk-job": hub_row(retry_of="job-from-sk"),
        "resume-of-sk-job": hub_row(execution={"resume": {"from_run": "job-from-sk"}}),
        "branch-of-sk-job": hub_row(execution={"branch": {"from_run": "job-from-sk"}}),
        # Programme steps.
        "step-steve": hub_row(execution={"programme": {"id": "prog-steve"}}),
        "step-from-steve-chat": hub_row(execution={"programme": {"id": "prog-steve-chat"}}),
        "step-from-sk-chat": hub_row(execution={"programme": {"id": "prog-sk-chat"}}),
        "step-of-gone-programme": hub_row(execution={"programme": {"id": "prog-gone"}}),
        # Schedules.
        "gardener": hub_row("routine", routine_id="gardener-routine"),
        "chat-schedule": hub_row("routine", routine_id="chat-routine"),
        "gone-schedule": hub_row("routine", routine_id="gone-routine"),
        # A client hub's fire from a chat it couldn't name.
        "owner-scoped-orphan": hub_row(execution={"owner_scoped": True}),
        "ui-run": hub_row(),
        # Started by a header-less POST to the hub (review A3).
        "from-chat-process": hub_row(execution={"caller_origin": {"kind": "chat", "why": "carries JARVIS_CHAT_TURN"}}),
        "from-sk-job-process": hub_row(execution={"caller_origin": {"kind": "runs", "runs": ["job-from-sk"]}}),
        "from-steve-job-process": hub_row(execution={"caller_origin": {"kind": "runs", "runs": ["job-from-steve"]}}),
        "from-garbled-origin": hub_row(execution={"caller_origin": "chat"}),
        # Schedules made from chats, named or not (review A4).
        "steve-chat-schedule": hub_row("routine", routine_id="steve-chat-routine"),
        "sk-chat-schedule": hub_row("routine", routine_id="sk-chat-routine"),
        "old-hub-schedule": hub_row("routine", routine_id="old-hub-routine"),
        "step-from-unnamed-chat": hub_row(execution={"programme": {"id": "prog-unnamed-chat"}}),
        "loop-a": hub_row(parent_id="loop-b"),
        "loop-b": hub_row(parent_id="loop-a"),
    },
    routines={"gardener-routine": {"created_from": None, "created_from_run": None},
              "chat-routine": {"created_from": "chat"},
              "steve-chat-routine": {"created_from": "chat", "created_from_run": "steve-tg"},
              "sk-chat-routine": {"created_from": "chat", "created_from_run": "sk-wa"},
              # A hub from before created_from was shown: can't be vouched for.
              "old-hub-routine": {"enabled": True}},
    programmes={"prog-steve": {"started_from": None},
                "prog-steve-chat": {"started_from": {"run_id": "steve-dc", "source": "dcbot"}},
                "prog-sk-chat": {"started_from": {"run_id": "sk-wa", "source": "wa"}},
                "prog-unnamed-chat": {"started_from": {"run_id": None, "source": "chat", "unplaced": True}}},
)

LOCAL = ("127.0.0.1", 50000)
NO_SLEEP = {"sleep": lambda s: None}


class RequesterTestCase(unittest.TestCase):
    def setUp(self):
        requester.clear_cache()
        self.sleep = patch("src.requester.time.sleep", lambda s: None)
        self.sleep.start()
        self.ports = patch.dict(os.environ, {"MEMORY_INDEX_FLEET_PORTS": "8080,8081,8084,8088",
                                             "MEMORY_INDEX_OWNER_PEERS": "100.64.0.7, 100.64.0.8/32"})
        self.ports.start()
        requester.clear_owner_peers()

    def tearDown(self):
        self.sleep.stop()
        self.ports.stop()
        requester.clear_cache()
        requester.clear_owner_peers()

    def decide(self, procs, headers=None, peer=LOCAL, hub=HUB):
        return requester.decide(peer, headers or {}, procs=procs, hub=hub)

    def turn(self, run_id, engine="claude.exe", daemon="python.exe", port=8081):
        """A chat daemon's engine for one turn, the daemon above it."""
        return chain((engine, {"JARVIS_RUN_ID": run_id, "JARVIS_CHAT_TURN": "telegram"}),
                     (daemon, {}), ("nssm.exe", {}), ("services.exe", {}),
                     listeners={daemon: port})

    def hub_run(self, run_id, engine="claude.exe"):
        return chain((engine, {"HUB_RUN_ID": run_id}), ("hubpython.exe", {}),
                     ("nssm.exe", {}), ("services.exe", {}), listeners={"hubpython.exe": 8084})


class TestChatTurns(RequesterTestCase):
    def test_the_owners_chat_turn_is_the_owners(self):
        v = self.decide(self.turn("steve-tg"))
        self.assertTrue(v.owner, v.why)

    def test_the_other_owners_turn_is_not(self):
        v = self.decide(self.turn("sk-wa", daemon="whatsapp-mcp.exe", port=8080))
        self.assertFalse(v.owner)
        self.assertIn("See Kiat", v.why)

    def test_the_peer_assistants_turn_is_not(self):
        self.assertFalse(self.decide(self.turn("peer-tg")).owner)

    def test_a_trusted_non_owners_turn_is_not(self):
        self.assertFalse(self.decide(self.turn("trusted-tg")).owner)

    def test_a_turn_with_no_recorded_sender_is_not(self):
        self.assertFalse(self.decide(self.turn("no-sender")).owner)
        self.assertFalse(self.decide(self.turn("unknown-bot")).owner)

    def test_a_wake_runs_as_whoever_fired_the_job(self):
        self.assertFalse(self.decide(self.turn("wake-for-sk")).owner)
        self.assertTrue(self.decide(self.turn("wake-for-steve")).owner)

    def test_codex_and_a_turns_own_shell_count_the_same(self):
        procs = chain(("curl.exe", {"JARVIS_RUN_ID": "sk-wa"}), ("bash.exe", {"JARVIS_RUN_ID": "sk-wa"}),
                      ("codex.exe", {"JARVIS_RUN_ID": "sk-wa", "JARVIS_CHAT_TURN": "whatsapp"}),
                      ("whatsapp-mcp.exe", {}), ("services.exe", {}),
                      listeners={"whatsapp-mcp.exe": 8080})
        self.assertFalse(self.decide(procs).owner)

    def test_a_turn_that_drops_its_run_id_but_keeps_the_marker_is_refused(self):
        procs = chain(("curl.exe", {"JARVIS_CHAT_TURN": "telegram"}), ("claude.exe", {}),
                      ("python.exe", {}), ("services.exe", {}), listeners={"python.exe": 8081})
        v = self.decide(procs)
        self.assertFalse(v.owner)
        self.assertIn("no run id", v.why)

    def test_a_turn_that_drops_every_marker_is_still_below_its_daemon(self):
        procs = chain(("curl.exe", {}), ("bash.exe", {}), ("python.exe", {}), ("services.exe", {}),
                      listeners={"python.exe": 8081})
        v = self.decide(procs)
        self.assertFalse(v.owner)
        self.assertIn("without a run id", v.why)


class TestHubRuns(RequesterTestCase):
    def test_a_job_fired_by_the_other_owner_is_not_the_owners(self):
        self.assertFalse(self.decide(self.hub_run("job-from-sk")).owner)
        self.assertFalse(self.decide(self.hub_run("subjob-from-sk")).owner)

    def test_a_job_fired_by_the_owner_is_the_owners(self):
        self.assertTrue(self.decide(self.hub_run("job-from-steve")).owner)

    def test_retries_resumes_and_branches_follow_the_run_they_carry_on(self):
        for tid in ("retry-of-sk-job", "resume-of-sk-job", "branch-of-sk-job"):
            self.assertFalse(self.decide(self.hub_run(tid)).owner, tid)

    def test_programme_steps_are_whoever_started_the_programme(self):
        self.assertTrue(self.decide(self.hub_run("step-steve")).owner)
        self.assertTrue(self.decide(self.hub_run("step-from-steve-chat")).owner)
        self.assertFalse(self.decide(self.hub_run("step-from-sk-chat")).owner)
        self.assertFalse(self.decide(self.hub_run("step-of-gone-programme")).owner)

    def test_the_gardener_and_owner_schedules_are_the_owners(self):
        self.assertTrue(self.decide(self.hub_run("gardener")).owner)
        self.assertTrue(self.decide(self.hub_run("ui-run")).owner)

    def test_a_schedule_a_chat_made_is_not(self):
        self.assertFalse(self.decide(self.hub_run("chat-schedule")).owner)
        self.assertFalse(self.decide(self.hub_run("gone-schedule")).owner)

    def test_a_schedule_is_whoever_made_it(self):
        self.assertTrue(self.decide(self.hub_run("steve-chat-schedule")).owner)
        self.assertFalse(self.decide(self.hub_run("sk-chat-schedule")).owner)
        v = self.decide(self.hub_run("old-hub-schedule"))
        self.assertFalse(v.owner)
        self.assertIn("doesn't say who made", v.why)

    def test_a_run_started_by_a_headerless_call_from_a_chat_process_is_not(self):
        for tid in ("from-chat-process", "from-garbled-origin"):
            v = self.decide(self.hub_run(tid))
            self.assertFalse(v.owner, tid)
        self.assertFalse(self.decide(self.hub_run("from-sk-job-process")).owner)
        self.assertTrue(self.decide(self.hub_run("from-steve-job-process")).owner)

    def test_a_programme_from_a_chat_the_hub_couldnt_name_is_not(self):
        self.assertFalse(self.decide(self.hub_run("step-from-unnamed-chat")).owner)

    def test_a_client_hubs_unnamed_chat_fire_is_not(self):
        self.assertFalse(self.decide(self.hub_run("owner-scoped-orphan")).owner)

    def test_a_run_the_hub_doesnt_know_is_refused(self):
        v = self.decide(self.hub_run("no-such-run"))
        self.assertFalse(v.owner)
        self.assertIn("doesn't know", v.why)

    def test_a_hub_that_cant_be_asked_is_refused(self):
        v = self.decide(self.hub_run("steve-tg"), hub=FakeHub(down=True))
        self.assertFalse(v.owner)
        self.assertIn("unreachable", v.why)

    def test_a_loop_of_parents_ends_refused(self):
        self.assertFalse(self.decide(self.hub_run("loop-a")).owner)


class TestSessionsAndProxies(RequesterTestCase):
    def test_the_owners_ssh_session_on_the_box(self):
        procs = chain(("claude.exe", {}), ("pwsh.exe", {}), ("sshd-session.exe", {}), ("sshd.exe", {}))
        v = self.decide(procs)
        self.assertTrue(v.owner, v.why)

    def test_the_owners_desktop_on_the_box(self):
        self.assertTrue(self.decide(chain(("claude.exe", {}), ("explorer.exe", {}))).owner)

    def test_the_tailscale_relay_is_judged_by_whom_it_relays(self):
        relay = chain(("tailscaled.exe", {}), ("services.exe", {}))
        # The owner's desktop through tailnet `serve`.
        v = self.decide(relay, {"X-Forwarded-For": "100.64.0.7"})
        self.assertTrue(v.owner, v.why)
        # The internet through the Funnel; a process on this box going round
        # through the relay; a relay that doesn't say.
        self.assertFalse(self.decide(relay, {"X-Forwarded-For": "160.79.106.161"}).owner)
        self.assertFalse(self.decide(relay, {"X-Forwarded-For": "127.0.0.1"}).owner)
        self.assertFalse(self.decide(relay).owner)
        # Only the entry the relay itself wrote counts.
        self.assertFalse(self.decide(relay, {"X-Forwarded-For": "100.64.0.7, 203.0.113.7"}).owner)

    def test_the_owners_other_machines_over_the_tailnet(self):
        v = self.decide(FakeProcs({}, {}, {}), peer=("100.64.0.7", 51000))
        self.assertTrue(v.owner, v.why)
        self.assertTrue(self.decide(FakeProcs({}, {}, {}), peer=("100.64.0.8", 51000)).owner)

    def test_other_machines_are_not_the_owners(self):
        # Another box on the tailnet (a client's admin, a peer box), the LAN,
        # the internet.
        for host in ("100.87.100.0", "192.168.50.20", "203.0.113.7"):
            v = self.decide(FakeProcs({}, {}, {}), peer=(host, 51000))
            self.assertFalse(v.owner, host)

    def test_no_owner_peers_means_no_remote_owner(self):
        with patch.dict(os.environ, {"MEMORY_INDEX_OWNER_PEERS": "none"}):
            requester.clear_owner_peers()
            self.assertFalse(self.decide(FakeProcs({}, {}, {}), peer=("100.64.0.7", 51000)).owner)

    def test_owner_peers_default_to_the_owners_untagged_tailnet_devices(self):
        status = {"Self": {"UserID": 1, "TailscaleIPs": ["100.64.0.1"]},
                  "Peer": {"a": {"UserID": 1, "TailscaleIPs": ["100.64.0.9", "fd7a::9"]},
                           "b": {"UserID": 2, "TailscaleIPs": ["100.64.0.10"]},
                           "c": {"UserID": 1, "Tags": ["tag:client"], "TailscaleIPs": ["100.64.0.11"]}}}
        tagged = {"Self": {"UserID": 3, "Tags": ["tag:client"]}, "Peer": status["Peer"]}
        env = {k: v for k, v in os.environ.items() if k != "MEMORY_INDEX_OWNER_PEERS"}
        for doc, want in ((status, ["100.64.0.9", "fd7a::9"]), (tagged, [])):
            run = MagicMock(return_value=types.SimpleNamespace(stdout=json.dumps(doc).encode()))
            with patch.dict(os.environ, env, clear=True), \
                    patch("src.requester._tailscale", return_value="tailscale"), \
                    patch("subprocess.run", run):
                requester.clear_owner_peers()
                self.assertEqual(requester._tailnet_owner_peers(), want)
                self.assertEqual(requester.is_owner_peer("100.64.0.9"), bool(want))
                self.assertFalse(requester.is_owner_peer("100.64.0.10"))
                self.assertFalse(requester.is_owner_peer("100.64.0.11"))

    def test_a_remote_header_can_only_narrow(self):
        v = self.decide(FakeProcs({}, {}, {}), {"X-Jarvis-Run": "sk-wa"}, peer=("100.64.0.7", 51000))
        self.assertFalse(v.owner)

    def test_a_forged_header_cant_borrow_the_owners_run(self):
        # The process names the other owner's turn; the header names Steve's.
        v = self.decide(self.turn("sk-wa"), {"X-Jarvis-Run": "steve-tg"})
        self.assertFalse(v.owner)

    def test_a_header_naming_someone_else_narrows_an_owner_session(self):
        procs = chain(("claude.exe", {}), ("sshd.exe", {}))
        self.assertFalse(self.decide(procs, {"X-Jarvis-Run": "sk-wa"}).owner)
        self.assertTrue(self.decide(procs, {"X-Jarvis-Run": "steve-tg"}).owner)

    def test_the_hubs_tool_gateway_must_name_the_run(self):
        hub = chain(("hubpython.exe", {}), ("nssm.exe", {}), ("services.exe", {}),
                    listeners={"hubpython.exe": 8084})
        self.assertFalse(self.decide(hub).owner)
        self.assertTrue(self.decide(hub, {"X-Jarvis-Run": "steve-tg"}).owner)
        self.assertFalse(self.decide(hub, {"X-Jarvis-Run": "sk-wa"}).owner)
        self.assertFalse(self.decide(hub, {"X-Jarvis-Run": "job-from-sk"}).owner)

    def test_a_chat_daemon_calling_for_someone_must_name_them(self):
        daemon = chain(("python.exe", {}), ("services.exe", {}), listeners={"python.exe": 8088})
        v = self.decide(daemon)
        self.assertFalse(v.owner)
        self.assertIn("doesn't name", v.why)

    def test_unplaceable_callers_are_refused(self):
        self.assertFalse(self.decide(FakeProcs({}, {}, {})).owner)            # nobody holds the port
        self.assertFalse(self.decide(chain(("curl.exe", {}))).owner)          # parent gone
        self.assertFalse(requester.decide(None, {}, procs=FakeProcs({}, {}, {}), hub=HUB).owner)

    def test_a_forged_forwarded_address_is_ignored_by_the_daemon(self):
        # uvicorn runs with proxy_headers off, so the peer stays the socket's
        # own and a local chat turn with X-Forwarded-For is still walked.
        import inspect
        from src import inflight
        self.assertIn("proxy_headers=False", inspect.getsource(inflight.serve_streamable_http))
        v = self.decide(self.turn("sk-wa"), {"X-Forwarded-For": "100.64.0.7"})
        self.assertFalse(v.owner)

    def test_a_detached_child_naming_an_owner_run_is_refused(self):
        # Its parent (another owner's turn) is gone; it names Steve's run
        # in its own environment (review A2).
        v = self.decide(chain(("python.exe", {"JARVIS_RUN_ID": "steve-tg"})))
        self.assertFalse(v.owner)
        self.assertIn("parent has gone", v.why)
        self.assertFalse(self.decide(chain(("python.exe", {"HUB_RUN_ID": "ui-run"}))).owner)

    def test_a_child_naming_an_owner_run_below_another_turn_is_refused(self):
        procs = chain(("python.exe", {"JARVIS_RUN_ID": "steve-tg", "JARVIS_CHAT_TURN": "telegram"}),
                      ("bash.exe", {"JARVIS_RUN_ID": "sk-wa", "JARVIS_CHAT_TURN": "telegram"}),
                      ("claude.exe", {"JARVIS_RUN_ID": "sk-wa", "JARVIS_CHAT_TURN": "telegram"}),
                      ("python.exe", {}), ("services.exe", {}), listeners={"python.exe": 8081})
        self.assertFalse(self.decide(procs).owner)

    def test_a_reused_parent_pid_is_not_a_parent(self):
        child = FakeProc(2, "curl.exe", {}, None, created=100.0)
        child._parent = FakeProc(1, "sshd.exe", {}, None, created=500.0)
        self.assertFalse(self.decide(FakeProcs({2: child, 1: child._parent}, {50000: 2}, {})).owner)

    def test_answers_are_cached_per_run(self):
        hub = FakeHub(triggers=dict(HUB.triggers), routines=HUB.routines, programmes=HUB.programmes)
        for _ in range(3):
            self.assertFalse(self.decide(self.turn("sk-wa"), hub=hub).owner)
        self.assertEqual(hub.asked.count("sk-wa"), 1)


# ---------------------------------------------------------------------------
# 2. Every tool, as a non-owner and as the owner
# ---------------------------------------------------------------------------

def _reset_state():
    import src.indexer.db as db_mod
    db_mod.reset()
    import src.indexer.store as store_mod
    store_mod._entities = {}
    store_mod._observations = {}
    store_mod._loaded = True
    import src.graph.manager as gm
    gm._graph = None
    gm._relations = {}
    import src.config as config_mod
    config_mod.VAULTS.clear()


class FakeCollection:
    """Enough of a Chroma collection for the store's writes and search's
    query: every vector at the same distance."""

    def __init__(self):
        self.ids: list[str] = []
        self.queries = 0

    def add(self, ids, **kw):
        self.ids.extend(ids)

    upsert = add

    def delete(self, ids=None, **kw):
        self.ids = [i for i in self.ids if i not in set(ids or [])]

    def count(self):
        return len(self.ids)

    def query(self, query_embeddings=None, n_results=10, where=None, include=None):
        self.queries += 1
        ids = self.ids[:n_results]
        return {"ids": [ids], "distances": [[0.1] * len(ids)],
                "metadatas": [[{} for _ in ids]], "documents": [["" for _ in ids]]}

    def get(self, **kw):
        return {"ids": [], "embeddings": [], "metadatas": []}


class ToolTestCase(unittest.TestCase):
    """A store with a work vault and a private one, a relation across them,
    driven through server.py's tools exactly as an MCP call would be."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.collections: dict[str, FakeCollection] = {}
        ef = MagicMock(side_effect=lambda texts: [[0.1] * 8 for _ in texts])
        coll = lambda name: self.collections.setdefault(name, FakeCollection())  # noqa: E731
        self.patches = [
            patch("src.config.DATA_DIR", Path(self.tmpdir)),
            patch("src.config.DB_FILE", Path(self.tmpdir) / "memory.db"),
            patch("src.config.CHROMA_DIR", Path(self.tmpdir) / "chroma"),
            patch("src.indexer.store.get_collection", side_effect=coll),
            patch("src.indexer.store.get_embedding_function", return_value=ef),
            patch("src.indexer.store.calibrate_collection", MagicMock()),
            patch("src.tools.search.get_collection", side_effect=coll),
            patch("src.tools.search._get_query_embeddings_with_guard", return_value=[[0.1] * 8]),
            patch("src.tools.search.spread_activation", MagicMock(return_value={})),
            patch("src.requester.owner_name", return_value="Steve"),
        ]
        for p in self.patches:
            p.start()
        _reset_state()
        # Modules that imported config.VAULTS hold whichever dict was current
        # then (other suites rebind it): point them all at today's.
        import src.config as config_mod
        for mod in ("src.tools.search", "src.tools.entities", "src.tools.librarian",
                    "src.tools.maintenance", "src.tools.temporal", "src.indexer.store"):
            p = patch(f"{mod}.VAULTS", config_mod.VAULTS)
            p.start()
            self.patches.append(p)
        import src.indexer.store as store_mod
        for name in ("_entities", "_observations"):
            p = patch(f"src.tools.temporal.{name}", getattr(store_mod, name))
            p.start()
            self.patches.append(p)
        from src.tools.search import _calibration_cache
        for v in ("work", "private"):
            _calibration_cache[v] = {"HIGH": 0.6, "MEDIUM": 1.0, "LOW": 1.4}
        import src.server as server
        self.s = server
        with access.as_verdict(OWNER):
            server.create_vault("work")
            server.create_vault("private")
            server.create_entity("Steve", "person", "work",
                                 ["Steve likes short replies"], source="test")
            server.create_entity("Steve health", "concept", "private",
                                 ["Steve has a knee injury"], source="test")
            server.create_entity("Steve", "person", "private",
                                 ["Steve's salary is secret"], source="test")
            server.create_relation("Steve health", "Steve", "related_to", vault="private")
            import src.indexer.store as store
            self.work_steve = store.get_entity_by_name("Steve", "work")
            self.health = store.get_entity_by_name("Steve health", "private")
            self.private_steve = store.get_entity_by_name("Steve", "private")
            server.create_relation(self.work_steve.id, self.health.id, "related_to")
            self.private_obs = store.get_observations(self.health.id)[0].id
            self.work_obs = store.get_observations(self.work_steve.id)[0].id
            import src.graph.manager as gm
            self.cross_rel = [r for r in gm.get_all_relations()
                              if r.from_entity == self.work_steve.id][0].id

    def tearDown(self):
        from tests.support import close_sqlite
        close_sqlite()
        for p in self.patches:
            p.stop()
        from src.tools.search import _calibration_cache
        _calibration_cache.clear()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def as_other(self):
        return access.as_verdict(NOT_OWNER)

    def as_owner(self):
        return access.as_verdict(OWNER)


SECRETS = ("knee", "salary", "Steve health")
REFUSED = "only for Steve"


class TestNonOwnerTools(ToolTestCase):
    def assertClean(self, text):
        for word in SECRETS:
            self.assertNotIn(word, text)

    def test_naming_the_vault_is_refused_everywhere(self):
        s = self.s
        calls = [
            lambda: s.create_entity("X", "concept", "private"),
            lambda: s.create_entity("X", "concept", " Private "),
            lambda: s.get_entity("Steve", vault="private"),
            lambda: s.update_entity("Steve", new_name="Y", vault="private"),
            lambda: s.reembed_entity("Steve", vault="private"),
            lambda: s.delete_entity("Steve", vault="private"),
            lambda: s.merge_entities("Steve", "Steve health", vault="private"),
            lambda: s.list_entities(vault="private"),
            lambda: s.add_observation("Steve", "x", vault="private"),
            lambda: s.add_observations("Steve", ["x"], vault="private"),
            lambda: s.create_relation("Steve", "Steve health", "related_to", vault="private"),
            lambda: s.search_memory("knee", vault="private"),
            lambda: s.get_neighbors("Steve", vault="private"),
            lambda: s.query_timeline(vault="private"),
            lambda: s.point_in_time("Steve", "2030-01-01", vault="private"),
            lambda: s.get_temporal_neighbors("Steve", vault="private"),
            lambda: s.analyze_graph(vault="private"),
            lambda: s.run_librarian(vault="private"),
            lambda: s.visualize_graph(vault="private"),
            lambda: s.create_vault("private"),
            lambda: s.delete_vault("private"),
            lambda: s.export_vault("private", output_path=self.tmpdir),
            lambda: s.vacuum_store(dry_run=True),
        ]
        with self.as_other():
            for i, call in enumerate(calls):
                out = call()
                self.assertIn(REFUSED, out, f"call {i}: {out}")
        with self.as_owner():
            self.assertIn("2 entities", s.list_vaults())
            self.assertIn("Steve has a knee injury", s.get_entity("Steve health", vault="private"))

    def test_lists_and_status_leave_it_out(self):
        with self.as_other():
            vaults = self.s.list_vaults()
            self.assertIn("work", vaults)
            self.assertNotIn("private", vaults)
            status = self.s.memory_status()
            self.assertNotIn("private", status)
            self.assertIn("Total entities: 1", status)
            self.assertClean(self.s.list_entities())
            self.assertIn("edges: 0", self.s.get_graph_summary().lower())
        with self.as_owner():
            self.assertIn("edges: 2", self.s.get_graph_summary().lower())
            self.assertIn("private", self.s.list_vaults())
            self.assertIn("Steve health", self.s.list_entities())

    def test_all_vault_search_silently_omits_it(self):
        with self.as_other():
            out = self.s.search_memory("Steve", output_format="json")
            self.assertClean(out)
            self.assertIn("short replies", out)
            self.assertEqual(self.collections["memory_private"].queries, 0)
        with self.as_owner():
            out = self.s.search_memory("Steve", output_format="json")
            self.assertIn("knee", out)

    def test_search_join_drops_a_private_hit_even_if_queried(self):
        # Belt and braces: a private vector that reached the join anyway.
        from src.tools import search
        with self.as_other():
            items = search._query_vault("private", [[0.1] * 8], 10, [], False)
        self.assertEqual(items, [])

    def test_reads_by_name_or_id_find_nothing(self):
        s = self.s
        with self.as_other():
            self.assertIn("not found", s.get_entity("Steve health").lower())
            self.assertIn("not found", s.get_entity(self.health.id).lower())
            # A name in both vaults resolves to the work one.
            self.assertIn("short replies", s.get_entity("Steve"))
            self.assertClean(s.get_entity("Steve", full=True))
            self.assertIn("not found", s.point_in_time(self.health.id, "2030-01-01").lower())
            self.assertClean(s.get_neighbors("Steve", vault="work"))
            self.assertClean(s.get_temporal_neighbors(self.work_steve.id))
            self.assertClean(s.query_timeline())
            self.assertClean(s.analyze_graph(output_format="json"))
            self.assertClean(s.reembed_status())

    def test_writes_by_name_or_id_find_nothing(self):
        s = self.s
        import src.indexer.store as store
        with self.as_other():
            self.assertIn("not found", s.add_observation(self.health.id, "leak").lower())
            self.assertIn("not found", s.add_observations(self.health.id, ["leak"]).lower())
            self.assertIn("not found", s.update_entity(self.health.id, new_name="Leak").lower())
            self.assertIn("not found", s.delete_entity(self.health.id).lower())
            self.assertIn("not found", s.reembed_entity(self.health.id).lower())
            self.assertIn("not found", s.merge_entities(self.health.id, "Steve").lower())
            self.assertIn("not found", s.delete_observation(self.private_obs).lower())
            self.assertIn("not found", s.create_relation("Steve", self.health.id, "related_to").lower())
            self.assertIn("not found", s.update_relation(self.cross_rel, context="x").lower())
            self.assertIn("not found", s.delete_relation(self.cross_rel).lower())
        with self.as_owner():
            # Nothing above landed.
            self.assertEqual(store.get_entity(self.health.id).name, "Steve health")
            obs = [o.content for o in store.get_observations(self.health.id)]
            self.assertEqual(obs, ["Steve has a knee injury"])
            import src.graph.manager as gm
            self.assertIsNotNone(gm.get_relation(self.cross_rel))

    def test_undelete_is_the_owners_alone(self):
        with self.as_owner():
            self.s.delete_observation(self.private_obs)
            self.s.delete_observation(self.work_obs)
        with self.as_other():
            # Not even in work: the notes moved out of work are deleted rows there.
            self.assertIn("Nothing was done", self.s.undelete_observation(self.private_obs))
            self.assertIn("Nothing was done", self.s.undelete_observation(self.work_obs))
        with self.as_owner():
            self.assertIn("restored", self.s.undelete_observation(self.private_obs).lower())
            self.assertIn("restored", self.s.undelete_observation(self.work_obs).lower())

    def test_export_and_import_are_the_owners_alone(self):
        with self.as_owner():
            self.assertIn("Exported", self.s.export_vault("work", output_path=self.tmpdir))
        archive = next(Path(self.tmpdir).glob("work_*.zip"))
        with self.as_other():
            self.assertIn("Nothing was done", self.s.export_vault("work", output_path=self.tmpdir))
            for target in ("", "work", "tmp"):
                self.assertIn("Nothing was done", self.s.import_vault(str(archive), vault=target))

    def test_import_into_it_is_refused(self):
        with self.as_owner():
            out = self.s.export_vault("private", output_path=self.tmpdir)
        archive = next(Path(self.tmpdir).glob("private_*.zip"))
        with self.as_other():
            self.assertIn(REFUSED, self.s.import_vault(str(archive)))
            self.assertIn(REFUSED, self.s.import_vault(str(archive), vault="private"))
        self.assertIn("private", out)

    def test_visualize_leaves_it_out(self):
        with patch("src.tools.visualize.os.startfile", create=True), \
                patch("src.tools.visualize.webbrowser.open"), \
                patch("src.tools.visualize.tempfile.gettempdir", return_value=self.tmpdir):
            with self.as_other():
                self.s.visualize_graph()
                html = (Path(self.tmpdir) / "memory-index" / "graph.html").read_text(encoding="utf-8")
                self.assertClean(html)
            with self.as_owner():
                self.s.visualize_graph()
                html = (Path(self.tmpdir) / "memory-index" / "graph.html").read_text(encoding="utf-8")
                self.assertIn("Steve health", html)

    def test_work_is_untouched_for_a_non_owner(self):
        with self.as_other():
            self.assertIn("Observation added", self.s.add_observation("Steve", "Prefers bullets", vault="work"))
            self.assertIn("Entity created", self.s.create_entity("Shared project", "project", "work"))
            self.assertIn("short replies", self.s.search_memory("Steve", vault="work"))


class TestOwnerTools(ToolTestCase):
    def test_the_owner_uses_it_like_any_vault(self):
        s = self.s
        with self.as_owner():
            self.assertIn("Observation added", s.add_observation("Steve health", "Physio on Fridays",
                                                                 vault="private", source="test"))
            self.assertIn("knee", s.search_memory("knee", vault="private"))
            self.assertIn("Steve health", s.list_entities(vault="private"))
            self.assertIn("Steve health", s.get_neighbors(self.work_steve.id))
            self.assertIn("knee", s.query_timeline(vault="private"))
            self.assertIn("Exported", s.export_vault("private", output_path=self.tmpdir))
            self.assertNotIn("only for Steve", s.vacuum_store(dry_run=True))


# ---------------------------------------------------------------------------
# 3. The HTTP daemon
# ---------------------------------------------------------------------------

HEADERS = {"accept": "application/json, text/event-stream", "content-type": "application/json"}


class TestHttpDaemon(ToolTestCase):
    def call(self, tool, arguments, verdict, client=("127.0.0.1", 50123)):
        """One MCP tools/call through the real streamable-HTTP app, the
        caller middleware and the inflight wrapper, `decide` answering
        `verdict` -- and the peer and headers it was asked about."""
        import httpx
        from mcp.server.transport_security import TransportSecuritySettings
        from src import inflight

        asked = []

        def decide(peer, headers, **kw):
            asked.append((peer, headers))
            return verdict

        async def run():
            mcp = self.s.mcp
            mcp.settings.streamable_http_path = "/mcp/keykeykeykey"
            mcp.settings.json_response = True
            mcp.settings.stateless_http = True
            mcp.settings.transport_security = TransportSecuritySettings(
                enable_dns_rebinding_protection=False)
            mcp._session_manager = None
            app = inflight.wrap(mcp, wrap_app=access.CallerMiddleware)
            transport = httpx.ASGITransport(app=app, client=client)
            async with mcp.session_manager.run(), \
                    httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as http:
                r = await http.post("/mcp/keykeykeykey", headers={**HEADERS, "X-Jarvis-Run": "abc12345"},
                                    json={"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                                          "params": {"name": tool, "arguments": arguments}})
                return r

        with patch("src.requester.decide", side_effect=decide):
            access.serving(True)
            try:
                r = asyncio.run(run())
            finally:
                access.serving(False)
        self.assertEqual(r.status_code, 200, r.text)
        text = "".join(c.get("text", "") for c in r.json()["result"]["content"])
        return text, asked

    def test_a_requests_caller_reaches_its_tool_call(self):
        text, asked = self.call("list_vaults", {}, NOT_OWNER)
        self.assertNotIn("private", text)
        self.assertEqual(asked[0][0], ("127.0.0.1", 50123))
        self.assertEqual(asked[0][1].get("x-jarvis-run"), "abc12345")
        self.assertEqual(len(asked), 1)   # decided once per request
        text, _ = self.call("list_vaults", {}, OWNER)
        self.assertIn("private", text)

    def test_search_over_http_omits_it_for_a_non_owner(self):
        text, _ = self.call("search_memory", {"query": "Steve"}, NOT_OWNER)
        self.assertNotIn("knee", text)
        self.assertIn("short replies", text)
        text, _ = self.call("search_memory", {"query": "knee", "vault": "private"}, NOT_OWNER)
        self.assertIn(REFUSED, text)

    def test_a_call_that_lost_its_request_is_refused_in_the_daemon(self):
        access.serving(True)
        try:
            self.assertFalse(access.private_allowed())
            self.assertNotIn("private", self.s.list_vaults())
            with access.system():
                self.assertTrue(access.private_allowed())
        finally:
            access.serving(False)
        self.assertTrue(access.private_allowed())   # stdio / tests: the owner

    def test_a_failing_decision_is_refused(self):
        req = access._Request(("127.0.0.1", 1), {})
        with patch("src.requester.decide", side_effect=RuntimeError("boom")):
            self.assertFalse(req.verdict().owner)


if __name__ == "__main__":
    unittest.main()
