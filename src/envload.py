"""The daemon's settings, in the fleet's one order.

memory-index read only its process environment (NSSM's AppEnvironmentExtra on
a fleet box), which made it the one service whose settings the hub's Settings
page could not keep in a file. It now loads, like every other daemon
(jarvis_core.settings in the chat daemons; the same few lines here, since
this package has no jarvis-core dependency):

  1. the process environment, which wins over both files;
  2. the box's instance.env (the superrepo root; for memory-index, whose
     checkout sits in the profile's Documents, that is INSTANCE_ROOT, else
     C:\\daemon-hub, else the checkout's parent), every key it sets: those are
     the box's and every service must hold them alike. A blank value is
     unset there;
  3. the checkout's own .env, for everything else.

load() returns one line naming what came from where, never a value; the
daemon logs it at start-up. JARVIS_INSTANCE_ENV names another instance file
(tests). Stdlib only.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import MutableMapping, Optional


def read_env_file(path: Path) -> dict[str, str]:
    """KEY=value lines; # comments and blanks skipped; one pair of matching
    quotes stripped; the last of a repeated key wins. {} when the file is
    missing or unreadable."""
    try:
        text = Path(path).read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError):
        return {}
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        out[key] = value
    return out


def instance_path(project_root: Path, env: MutableMapping[str, str]) -> Path:
    """JARVIS_INSTANCE_ENV, else <INSTANCE_ROOT>\\instance.env, else
    C:\\daemon-hub\\instance.env when that folder exists, else beside the checkout."""
    named = (env.get("JARVIS_INSTANCE_ENV") or "").strip()
    if named:
        return Path(named)
    root = (env.get("INSTANCE_ROOT") or "").strip()
    if root:
        return Path(root) / "instance.env"
    default = Path(r"C:\daemon-hub")
    if default.is_dir():
        return default / "instance.env"
    return Path(project_root).resolve().parent / "instance.env"


def load(project_root: Path, env: Optional[MutableMapping[str, str]] = None,
         instance_file: Optional[Path] = None) -> str:
    """Put the checkout's .env and the box's instance.env into `env`
    (os.environ) in the order above; the line to log."""
    env = os.environ if env is None else env
    root = Path(project_root)
    dotenv = read_env_file(root / ".env")
    inst_path = Path(instance_file) if instance_file else instance_path(root, env)
    instance = {k: v for k, v in read_env_file(inst_path).items() if v.strip()}
    had = set(env)
    from_instance, from_env_file, overrides = [], [], []
    for key, value in instance.items():
        if key in had:
            continue
        env[key] = value
        from_instance.append(key)
        if key in dotenv and dotenv[key] != value:
            overrides.append(key)
    for key, value in dotenv.items():
        if key in had or key in instance:
            continue
        env[key] = value
        from_env_file.append(key)
    line = f"settings: {len(from_env_file)} from .env"
    if inst_path.is_file():
        line += f"; {len(from_instance)} from {inst_path} ({', '.join(from_instance) or 'none'})"
    if overrides:
        line += f". instance.env wins over the .env's {', '.join(overrides)}: the .env is stale"
    return line
