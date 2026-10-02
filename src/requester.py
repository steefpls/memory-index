"""Who an HTTP call to memory-index is for: the box's owner, or not.

src/access.py asks `decide` once per request, and only the owner may use
the `private` vault. Every rule below fails closed: what can't be vouched
for as the owner's is not the owner's.

1. Where the call comes from.

   - No peer address: refused.
   - A peer that isn't this machine (the owner's own desktop or laptop on
     the tailnet; a client box is fenced off from everyone else's): the
     owner.
   - A peer on this machine: the process holding the connection, and the
     processes above it, say who it is (psutil). memory-index runs as
     LocalSystem, so it can read their environments:

       JARVIS_RUN_ID    a chat turn: the hub trigger id a chat daemon gives
                        every engine it starts (jarvis-core runner.py,
                        whatsapp-mcp claude/trigger.go);
       HUB_RUN_ID       a hub run: set by the hub on every run it starts
                        (orchestrator-hub claude_subproc.py);
       JARVIS_CHAT_TURN a chat turn's marker. Seen without JARVIS_RUN_ID
                        it is a turn nobody can name: refused.

     The walk stops at the first fleet service (a process listening on one
     of FLEET_PORTS: the hub, a chat daemon...) or at a Windows session or
     service root (sshd, explorer, services: the owner on the box, the
     box's own services such as the Funnel relay). A fleet service that
     holds the connection ITSELF is calling for someone else (the hub's
     OpenCode tool gateway, discord-mcp's /whois): it must name the run in
     an X-Jarvis-Run header, or it is refused. A chain that breaks or runs
     too long without a name is refused.

   An X-Jarvis-Run header from anyone else can only narrow: the run it
   names must be the owner's too. Trigger ids are no secret (the hub shows
   them), so a header on its own never vouches for anything.

2. Whose each named run is, from the hub's records (GET /api/triggers/{id}):

   - a chat turn (any source but hub/routine): the owner's only when the
     message that started it was the owner's (sender_owner). A peer turn,
     a trusted non-owner's, the other owner's, a turn with no sender: no.
     A wake runs as the sender whose turn fired the job, so the same rule
     covers it.
   - any run it came from must be the owner's too: parent_id (run_fire from
     a run or a chat), retry_of, a resume's or a branch's from_run, the chat
     run a programme was started from.
   - a fire of a routine a chat turn made (created_from 'chat'): no -- the
     hub doesn't record which chat. A fire the hub marked owner_scoped with
     no parent (a client's chat it couldn't name): no.
   - a run the hub doesn't know, or a hub that can't be asked: no.
   - anything else (Steve's or the owner's programmes and schedules, the
     gardener, the hub's own runs, runs started from the hub UI): yes.
"""
from __future__ import annotations

import ipaddress
import json
import logging
import os
import socket
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from src.access import Verdict

logger = logging.getLogger(__name__)

RUN_HEADER = "x-jarvis-run"
RUN_ENV = "JARVIS_RUN_ID"
HUB_RUN_ENV = "HUB_RUN_ID"
CHAT_ENV = "JARVIS_CHAT_TURN"

# The hub's own runs and schedules; every other source is a chat surface.
HUB_SOURCES = frozenset({"hub", "routine"})

# Image names whose chain is a person's session on the box or the box's own
# services (orchestrator-hub caller_process.ADMIN_ROOTS).
ROOTS = frozenset({
    "sshd.exe", "sshd", "sshd-session.exe", "explorer.exe", "services.exe", "wininit.exe",
    "winlogon.exe", "svchost.exe", "csrss.exe", "smss.exe", "system", "system idle process",
    "userinit.exe", "launchd", "systemd", "init",
})
_MAX_HOPS = 24
_MAX_RUN_HOPS = 6
_CACHE_S = 300.0
_NOT_FOUND_RETRY_S = 1.5


_owner: list[str] = []


def owner_name() -> str:
    """The box's owner, for messages: OWNER_NAME from the environment or the
    box's instance.env (a client box), else Steve (steef-server has none)."""
    if not _owner:
        name = (os.environ.get("OWNER_NAME") or "").strip()
        if not name:
            root = (os.environ.get("INSTANCE_ROOT") or "").strip() or r"C:\daemon-hub"
            try:
                text = open(os.path.join(root, "instance.env"), encoding="utf-8-sig").read()
            except OSError:
                text = ""
            for line in text.splitlines():
                key, sep, value = line.strip().partition("=")
                if sep and key.strip() == "OWNER_NAME":
                    name = value.strip()
        _owner.append(name or "Steve")
    return _owner[0]


def hub_url() -> str:
    return (os.environ.get("MEMORY_INDEX_HUB_URL") or "http://127.0.0.1:8084").strip().rstrip("/")


def fleet_ports() -> frozenset[int]:
    """Ports of the fleet services that may call memory-index for someone
    else: whatsapp 8080, telegram 8081, hub 8084, daemon-hub-mcp 8085,
    dataprocessor 8087, discord 8088 (and Google Workspace 8083)."""
    raw = os.environ.get("MEMORY_INDEX_FLEET_PORTS") or "8080,8081,8083,8084,8085,8087,8088"
    out = set()
    for part in raw.split(","):
        try:
            out.add(int(part.strip()))
        except ValueError:
            continue
    return frozenset(out)


# --- 1. the calling process -------------------------------------------------

@dataclass
class Origin:
    """What the walk up from the calling process found."""
    runs: list[tuple[str, str]] = field(default_factory=list)   # (trigger id, where)
    refused: str | None = None                                   # why, when unnameable
    proxy: str | None = None                                     # the service holding the connection
    root: str | None = None


def is_local(host: str) -> bool:
    host = _norm(host)
    if not host:
        return False
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return host == "localhost"
    if ip.is_loopback:
        return True
    return host in _own_addresses()


def _norm(ip: str) -> str:
    ip = (ip or "").strip().strip("[]").lower()
    if ip.startswith("::ffff:"):
        ip = ip[7:]
    return ip.split("%", 1)[0]


_own: tuple[float, frozenset[str]] = (0.0, frozenset())


def _own_addresses() -> frozenset[str]:
    global _own
    at, addrs = _own
    if addrs and time.monotonic() - at < 300:
        return addrs
    found = {"127.0.0.1", "::1"}
    try:
        import psutil
        for entries in psutil.net_if_addrs().values():
            for a in entries:
                if a.address and ("." in a.address or ":" in a.address):
                    found.add(_norm(a.address))
    except Exception:  # noqa: BLE001
        try:
            found.update(_norm(i[4][0]) for i in socket.getaddrinfo(socket.gethostname(), None))
        except OSError:
            pass
    _own = (time.monotonic(), frozenset(found))
    return _own[1]


def _safe(fn: Callable[[], Any], default: Any) -> Any:
    try:
        return fn()
    except Exception:  # noqa: BLE001 -- psutil.Error, a process gone mid-walk
        return default


class Processes:
    """The psutil calls the walk makes, in one place so tests can fake them."""

    def __init__(self) -> None:
        import psutil
        self._psutil = psutil
        self._conns = list(psutil.net_connections("tcp"))

    def pid_at(self, ip: str, port: int) -> int | None:
        ip = _norm(ip)
        for c in self._conns:
            la = c.laddr
            if la and c.pid and la.port == port and _norm(la.ip) == ip:
                return int(c.pid)
        return None

    def services(self, ports: frozenset[int]) -> dict[int, int]:
        """{pid: port} of the processes listening on a fleet port."""
        out: dict[int, int] = {}
        for c in self._conns:
            la = c.laddr
            if la and c.pid and la.port in ports and str(getattr(c, "status", "")).upper() == "LISTEN":
                out[int(c.pid)] = la.port
        return out

    def proc(self, pid: int) -> Any:
        return self._psutil.Process(pid)


def walk(pid: int | None, headers: Mapping[str, str], procs: Any,
         ports: frozenset[int] | None = None) -> Origin:
    """What the process holding the connection, and those above it, say."""
    out = Origin()
    if not pid:
        out.refused = "no process holds the calling connection"
        return out
    services = procs.services(ports if ports is not None else fleet_ports())
    proc = _safe(lambda: procs.proc(pid), None)
    if proc is None:
        out.refused = f"process {pid} is gone"
        return out
    for hop in range(_MAX_HOPS):
        name = (_safe(proc.name, "") or "").lower()
        env = _safe(proc.environ, None)
        if env is None:
            env = {}
        run = (env.get(RUN_ENV) or "").strip()
        hub_run = (env.get(HUB_RUN_ENV) or "").strip()
        if run:
            out.runs.append((run, f"{RUN_ENV} on {name or proc.pid}"))
        if hub_run:
            out.runs.append((hub_run, f"{HUB_RUN_ENV} on {name or proc.pid}"))
        if (env.get(CHAT_ENV) or "").strip() and not run:
            out.refused = f"{name or proc.pid} is a chat turn that carries no run id"
            return out
        if proc.pid in services:
            if hop == 0:
                # A fleet service calling for someone: it must say whom.
                out.proxy = f"{name or proc.pid} (port {services[proc.pid]})"
                named = (headers.get(RUN_HEADER) or "").strip()
                if named:
                    out.runs.append((named, f"X-Jarvis-Run from {out.proxy}"))
                else:
                    out.refused = f"{out.proxy} calls for someone it doesn't name"
                return out
            if not out.runs:
                out.refused = f"started by {name or proc.pid} (port {services[proc.pid]}) without a run id"
            return out
        if name in ROOTS:
            out.root = name
            return out
        parent = _safe(proc.parent, None)
        # A reused pid: a "parent" younger than its child isn't one.
        if parent is not None and _safe(parent.create_time, 0.0) > _safe(proc.create_time, 0.0) + 1:
            parent = None
        if parent is None:
            if not out.runs:
                out.refused = f"{name or proc.pid}'s parent has gone"
            return out
        proc = parent
    if not out.runs:
        out.refused = "too many parents to follow"
    return out


# --- 2. the hub's records ----------------------------------------------------

class HubError(RuntimeError):
    """The hub couldn't be asked."""


def _get(path: str, timeout: float = 5.0) -> dict | None:
    """GET a hub JSON document; None on 404."""
    url = hub_url() + path
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"accept": "application/json"}),
                                    timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise HubError(f"hub answered {e.code} for {path}") from e
    except (OSError, ValueError) as e:
        raise HubError(f"hub unreachable for {path}: {e!r}") from e


class Hub:
    """The hub's records, read over its local API."""

    def trigger(self, tid: str) -> dict | None:
        # events=0: a newer hub leaves the transcript out; an older one
        # ignores it and sends everything, which still works.
        d = _get(f"/api/triggers/{tid}?events=0")
        if d is None:
            return None
        return d.get("trigger") if isinstance(d.get("trigger"), dict) else d

    def routine(self, rid: str) -> dict | None:
        return _get(f"/api/routines/{rid}")

    def programme(self, pid: str) -> dict | None:
        d = _get(f"/api/programmes/{pid}")
        if d is None:
            return None
        return d.get("programme") if isinstance(d.get("programme"), dict) else d


_cache: dict[str, tuple[float, bool, str]] = {}
_cache_lock = threading.Lock()


def _cached(tid: str) -> tuple[bool, str] | None:
    with _cache_lock:
        hit = _cache.get(tid)
        if hit and time.monotonic() - hit[0] < _CACHE_S:
            return hit[1], hit[2]
    return None


def _remember(tid: str, owner: bool, why: str) -> None:
    with _cache_lock:
        if len(_cache) > 2000:
            _cache.clear()
        _cache[tid] = (time.monotonic(), owner, why)


def clear_cache() -> None:
    with _cache_lock:
        _cache.clear()


def run_is_owners(tid: str, hub: Any, hops: int = _MAX_RUN_HOPS,
                  sleep: Callable[[float], None] = time.sleep) -> tuple[bool, str]:
    """(is run `tid` the owner's, why). Raises HubError when the hub can't
    be asked (the caller refuses)."""
    tid = (tid or "").strip()
    if not tid:
        return False, "an empty run id"
    hit = _cached(tid)
    if hit is not None:
        return hit
    if hops <= 0:
        return False, f"run {tid[:8]} sits under too many others to follow"
    row = hub.trigger(tid)
    if row is None and hops == _MAX_RUN_HOPS:
        # A chat turn's first tool call can race its daemon's stream header.
        sleep(_NOT_FOUND_RETRY_S)
        row = hub.trigger(tid)
    if row is None:
        return False, f"the hub doesn't know run {tid[:8]}"
    owner, why = _row_is_owners(tid, row, hub, hops, sleep)
    _remember(tid, owner, why)
    return owner, why


def _row_is_owners(tid: str, row: Mapping[str, Any], hub: Any, hops: int,
                   sleep: Callable[[float], None]) -> tuple[bool, str]:
    short = tid[:8]
    source = str(row.get("source") or "hub").strip().lower()
    ex = row.get("execution") if isinstance(row.get("execution"), dict) else {}
    if source not in HUB_SOURCES:
        if row.get("sender_owner") is not True:
            who = row.get("sender_name") or row.get("sender_id") or "nobody the hub recorded"
            return False, f"chat run {short} ({source}) was asked by {who}, not {owner_name()}"
    links: list[str] = []
    for linked in (row.get("parent_id"), ex.get("parent_id"), row.get("retry_of"),
                   (ex.get("resume") or {}).get("from_run") if isinstance(ex.get("resume"), dict) else None,
                   (ex.get("branch") or {}).get("from_run") if isinstance(ex.get("branch"), dict) else None):
        if isinstance(linked, str) and linked.strip() and linked.strip() != tid:
            links.append(linked.strip())
    if source in HUB_SOURCES and ex.get("owner_scoped") and not links:
        return False, f"run {short} was fired by a chat the hub couldn't name"
    routine_id = row.get("routine_id")
    if routine_id:
        routine = hub.routine(str(routine_id))
        if routine is None:
            return False, f"run {short}'s schedule {str(routine_id)[:8]} is gone"
        if str(routine.get("created_from") or "").lower() == "chat":
            return False, f"run {short}'s schedule was made from a chat"
    prog = ex.get("programme") if isinstance(ex.get("programme"), dict) else None
    if prog and prog.get("id"):
        p = hub.programme(str(prog["id"]))
        if p is None:
            return False, f"run {short}'s programme {str(prog['id'])[:8]} is gone"
        started = p.get("started_from") if isinstance(p.get("started_from"), dict) else None
        if started and started.get("run_id"):
            links.append(str(started["run_id"]))
    for linked in dict.fromkeys(links):
        owner, why = run_is_owners(linked, hub, hops - 1, sleep)
        if not owner:
            return False, f"run {short} came from run {linked[:8]}: {why}"
    if source not in HUB_SOURCES:
        return True, f"chat run {short} ({source}) was asked by {owner_name()}"
    return True, f"run {short} ({source}) was started by {owner_name()} or the box"


# --- the decision -------------------------------------------------------------

def decide(peer: tuple[str, int] | None, headers: Mapping[str, str], *,
           procs: Any = None, hub: Any = None) -> Verdict:
    """Is this request the owner's? (See the module doc.)"""
    headers = {str(k).lower(): str(v) for k, v in headers.items()}
    if not peer or not peer[0]:
        return Verdict(False, "no peer address")
    host, port = peer
    named = (headers.get(RUN_HEADER) or "").strip()
    if not is_local(host):
        if named:
            return _runs_verdict([(named, "X-Jarvis-Run")], hub)
        return Verdict(True, f"a session from {host}, another of the owner's machines")
    procs = procs if procs is not None else Processes()
    origin = walk(procs.pid_at(host, port), headers, procs)
    if origin.refused:
        return Verdict(False, origin.refused)
    runs = list(origin.runs)
    if named and not origin.proxy:
        runs.append((named, "X-Jarvis-Run"))
    if not runs:
        return Verdict(True, f"a session on this box ({origin.root or 'no run'})")
    return _runs_verdict(runs, hub)


def _runs_verdict(runs: list[tuple[str, str]], hub: Any) -> Verdict:
    hub = hub if hub is not None else Hub()
    whys = []
    first: dict[str, str] = {}
    for tid, where in runs:
        first.setdefault(tid, where)
    for tid, where in first.items():
        try:
            owner, why = run_is_owners(tid, hub)
        except HubError as e:
            return Verdict(False, f"{where}: {e}")
        if not owner:
            return Verdict(False, f"{where}: {why}")
        whys.append(why)
    return Verdict(True, "; ".join(whys))
