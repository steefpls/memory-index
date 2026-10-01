"""Tests for the Drive backup's settings (scripts/backup_to_drive.py).

Unset, every setting is steef-server's; another instance names its own
vault, Google account and profile.
"""

import importlib.util
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Every test names its instance.env, so the box the suite runs on (a client
# box has one at its fleet root) never changes what it sees.
NOWHERE = PROJECT_ROOT / "tests" / "no-such-instance.env"


def _load_backup():
    """scripts/ is not a package, so load the script by path."""
    path = PROJECT_ROOT / "scripts" / "backup_to_drive.py"
    spec = importlib.util.spec_from_file_location("backup_to_drive", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BackupSettingsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backup = _load_backup()

    def test_defaults_are_steef_servers(self):
        vault, token = self.backup.backup_settings({}, NOWHERE)
        self.assertEqual(vault, "work")
        self.assertEqual(
            token,
            Path(r"C:\Users\steve") / ".google_workspace_mcp" / "credentials"
            / "steven.koe80@gmail.com.json",
        )

    def test_blank_values_fall_back_to_defaults(self):
        vault, token = self.backup.backup_settings(
            {"MEMORY_BACKUP_VAULT": " ", "OWNER_GOOGLE_EMAIL": "", "INSTANCE_PROFILE": ""}, NOWHERE
        )
        self.assertEqual((vault, token), self.backup.backup_settings({}, NOWHERE))

    def test_another_instance(self):
        vault, token = self.backup.backup_settings({
            "MEMORY_BACKUP_VAULT": "sk",
            "OWNER_GOOGLE_EMAIL": "sk@example.com",
            "INSTANCE_PROFILE": r"D:\Users\sk",
        }, NOWHERE)
        self.assertEqual(vault, "sk")
        self.assertEqual(
            token,
            Path(r"D:\Users\sk") / ".google_workspace_mcp" / "credentials" / "sk@example.com.json",
        )

    def _instance_env(self, text):
        path = Path(tempfile.mkdtemp()) / "instance.env"
        path.write_text(text, encoding="utf-8")
        return path

    def test_a_client_box_reads_its_instance_env(self):
        path = self._instance_env(
            "# Friday\nINSTANCE_ROLE=client\nOWNER_GOOGLE_EMAIL=sk@example.com\n"
            "INSTANCE_PROFILE=C:\\Users\\seeki\nMEMORY_VAULT=work\n")
        vault, token = self.backup.backup_settings({}, path)
        self.assertEqual(vault, "work")
        self.assertEqual(token, Path(r"C:\Users\seeki") / ".google_workspace_mcp" / "credentials"
                         / "sk@example.com.json")
        # The environment still wins over the file.
        _, token = self.backup.backup_settings({"OWNER_GOOGLE_EMAIL": "x@example.com"}, path)
        self.assertEqual(token.name, "x@example.com.json")

    def test_a_client_box_never_falls_back_to_steef_server(self):
        path = self._instance_env("INSTANCE_ROLE=client\nINSTANCE_PROFILE=C:\\Users\\seeki\n")
        with self.assertRaisesRegex(RuntimeError, "OWNER_GOOGLE_EMAIL"):
            self.backup.backup_settings({}, path)

    def test_a_dev_instance_env_still_defaults_to_steef_server(self):
        path = self._instance_env("INSTANCE_ROLE=dev\n")
        self.assertEqual(self.backup.backup_settings({}, path), self.backup.backup_settings({}, NOWHERE))

    def test_a_vault_not_created_yet_is_nothing_to_back_up(self):
        backup = self.backup
        calls = []
        saved = (backup.tool_export_vault, backup.setup_logging, backup._drive_phase)
        try:
            backup.tool_export_vault = lambda vault, path: f"Vault '{vault}' not found."
            backup.setup_logging = lambda: None
            backup._drive_phase = lambda path: calls.append(path)
            self.assertIsNone(backup.export_vault_local())
            self.assertEqual(backup.main(), 0)
            self.assertEqual(calls, [])
        finally:
            backup.tool_export_vault, backup.setup_logging, backup._drive_phase = saved

    def test_import_writes_no_log_file(self):
        # Logging is set up in main(), so loading the module (as this test
        # does) must not attach a file handler to the root logger.
        import logging

        log_file = str(self.backup.LOG_FILE)
        handlers = [
            h for h in logging.getLogger().handlers
            if getattr(h, "baseFilename", None) == log_file
        ]
        self.assertEqual(handlers, [])


if __name__ == "__main__":
    unittest.main()
