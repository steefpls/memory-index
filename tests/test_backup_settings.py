"""Tests for the Drive backup's settings (scripts/backup_to_drive.py).

Unset, every setting is steef-server's; another instance names its own
vault, Google account and profile.
"""

import importlib.util
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


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
        vault, token = self.backup.backup_settings({})
        self.assertEqual(vault, "work")
        self.assertEqual(
            token,
            Path(r"C:\Users\steve") / ".google_workspace_mcp" / "credentials"
            / "steven.koe80@gmail.com.json",
        )

    def test_blank_values_fall_back_to_defaults(self):
        vault, token = self.backup.backup_settings(
            {"MEMORY_BACKUP_VAULT": " ", "OWNER_GOOGLE_EMAIL": "", "INSTANCE_PROFILE": ""}
        )
        self.assertEqual((vault, token), self.backup.backup_settings({}))

    def test_another_instance(self):
        vault, token = self.backup.backup_settings({
            "MEMORY_BACKUP_VAULT": "sk",
            "OWNER_GOOGLE_EMAIL": "sk@example.com",
            "INSTANCE_PROFILE": r"D:\Users\sk",
        })
        self.assertEqual(vault, "sk")
        self.assertEqual(
            token,
            Path(r"D:\Users\sk") / ".google_workspace_mcp" / "credentials" / "sk@example.com.json",
        )

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
