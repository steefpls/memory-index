"""src/envload.py: the environment, then instance.env, then the .env."""
import tempfile
from pathlib import Path

from src import envload


def test_env_then_instance_then_dotenv():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "memory-index"
        root.mkdir()
        (root / ".env").write_text("MCP_PORT=8082\nOWNER_NAME=Steve\nMEMORY_INDEX_EMBED_DEVICE=cuda\n", encoding="utf-8")
        inst = Path(tmp) / "instance.env"
        inst.write_text("OWNER_NAME=Alex\nINSTANCE_HOST=\nINSTANCE_TZ=Europe/London\n", encoding="utf-8")
        env = {"MCP_PORT": "9000"}
        line = envload.load(root, env, instance_file=inst)
        assert env == {"MCP_PORT": "9000", "OWNER_NAME": "Alex", "INSTANCE_TZ": "Europe/London",
                       "MEMORY_INDEX_EMBED_DEVICE": "cuda"}
        assert "instance.env wins over the .env's OWNER_NAME" in line
        assert "Alex" not in line


def test_no_files_changes_nothing():
    with tempfile.TemporaryDirectory() as tmp:
        env = {"A": "1"}
        line = envload.load(Path(tmp) / "memory-index", env, instance_file=Path(tmp) / "none.env")
        assert env == {"A": "1"}
        assert line == "settings: 0 from .env"


def test_instance_path_is_named_or_the_boxs_root():
    root = Path("D:/x/memory-index")
    assert envload.instance_path(root, {"JARVIS_INSTANCE_ENV": "D:/y/i.env"}) == Path("D:/y/i.env")
    assert envload.instance_path(root, {"INSTANCE_ROOT": r"D:\hub"}) == Path(r"D:\hub") / "instance.env"
