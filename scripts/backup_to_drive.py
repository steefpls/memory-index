"""Daily memory-index work-vault backup to Google Drive.

Runs on steef-server via Windows Task Scheduler. Calls
`tool_export_vault('work', ...)` to produce a date-stamped zip in
`data/exports/`, then uploads to a Drive folder named
`memory-index-backups` using the OAuth token shared with
google_workspace_mcp.

Local copies are pruned to LOCAL_RETENTION_DAYS; Drive copies are
pruned to DRIVE_RETENTION_DAYS.

Designed to run as the SYSTEM account from Task Scheduler with
USERPROFILE pointed at C:\\Users\\steve in the task action's env vars.
The OAuth token path is hardcoded so SYSTEM-execution doesn't read
from Windows\\System32\\config\\systemprofile by mistake.
"""

import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.tools.portability import tool_export_vault  # noqa: E402

VAULT = "work"
EXPORTS_DIR = PROJECT_ROOT / "data" / "exports"
LOG_FILE = PROJECT_ROOT / "data" / "backup.log"
LOCAL_RETENTION_DAYS = 7
DRIVE_RETENTION_DAYS = 30
DRIVE_FOLDER_NAME = "memory-index-backups"

# Hardcoded — Path.home() resolves wrong under SYSTEM (points to
# C:\Windows\System32\config\systemprofile). Server is one specific
# host so a hardcoded path is correct here.
OAUTH_TOKEN_PATH = Path(
    r"C:\Users\steve\.google_workspace_mcp\credentials\steven.koe80@gmail.com.json"
)

LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stderr),
    ],
)
log = logging.getLogger("backup_to_drive")


def export_vault_local() -> Path:
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_path = EXPORTS_DIR / f"{VAULT}_{today}.zip"
    result = tool_export_vault(VAULT, str(out_path))
    log.info("Export: %s", " ".join(result.split()))
    if not out_path.exists():
        raise RuntimeError(f"Export reported success but file missing: {out_path}")
    return out_path


def get_drive_service():
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build

    if not OAUTH_TOKEN_PATH.exists():
        raise RuntimeError(f"OAuth token not found at {OAUTH_TOKEN_PATH}")

    with OAUTH_TOKEN_PATH.open("r", encoding="utf-8") as f:
        token_data = json.load(f)

    creds = Credentials.from_authorized_user_info(token_data, scopes=token_data.get("scopes"))
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        # Save the refreshed token back so the next run / google_workspace_mcp
        # picks up the new access token without re-auth.
        with OAUTH_TOKEN_PATH.open("w", encoding="utf-8") as f:
            f.write(creds.to_json())
        log.info("Refreshed OAuth access token")

    return build("drive", "v3", credentials=creds, cache_discovery=False)


def ensure_drive_folder(service) -> str:
    query = (
        f"name = '{DRIVE_FOLDER_NAME}' and "
        "mimeType = 'application/vnd.google-apps.folder' and "
        "trashed = false"
    )
    res = service.files().list(q=query, fields="files(id, name)", pageSize=1).execute()
    files = res.get("files", [])
    if files:
        return files[0]["id"]
    meta = {"name": DRIVE_FOLDER_NAME, "mimeType": "application/vnd.google-apps.folder"}
    folder = service.files().create(body=meta, fields="id").execute()
    log.info("Created Drive folder %s (id=%s)", DRIVE_FOLDER_NAME, folder["id"])
    return folder["id"]


def upload_to_drive(service, folder_id: str, local_path: Path) -> str:
    from googleapiclient.http import MediaFileUpload

    meta = {"name": local_path.name, "parents": [folder_id]}
    media = MediaFileUpload(str(local_path), mimetype="application/zip", resumable=False)
    f = service.files().create(body=meta, media_body=media, fields="id, name").execute()
    log.info("Uploaded %s (id=%s, %d bytes)", f["name"], f["id"], local_path.stat().st_size)
    return f["id"]


def upload_oauth_snapshot(service, folder_id: str) -> str:
    """Upload a date-stamped copy of the OAuth token to the same Drive folder.

    Loss of this token only costs a 5-min re-auth on the server, but having a
    snapshot means a corrupted token file can be restored without re-running
    the OAuth flow.
    """
    from googleapiclient.http import MediaFileUpload

    if not OAUTH_TOKEN_PATH.exists():
        log.warning("OAuth token missing at %s, skipping snapshot", OAUTH_TOKEN_PATH)
        return ""

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    drive_name = f"oauth_{OAUTH_TOKEN_PATH.stem}_{today}.json"
    meta = {"name": drive_name, "parents": [folder_id]}
    media = MediaFileUpload(str(OAUTH_TOKEN_PATH), mimetype="application/json", resumable=False)
    f = service.files().create(body=meta, media_body=media, fields="id, name").execute()
    log.info("Uploaded %s (id=%s, %d bytes)", f["name"], f["id"], OAUTH_TOKEN_PATH.stat().st_size)
    return f["id"]


def prune_drive_backups(service, folder_id: str) -> int:
    cutoff = datetime.now(timezone.utc) - timedelta(days=DRIVE_RETENTION_DAYS)
    cutoff_iso = cutoff.strftime("%Y-%m-%dT%H:%M:%SZ")
    query = f"'{folder_id}' in parents and trashed = false and modifiedTime < '{cutoff_iso}'"
    res = service.files().list(q=query, fields="files(id, name)", pageSize=200).execute()
    files = res.get("files", [])
    for f in files:
        try:
            service.files().delete(fileId=f["id"]).execute()
            log.info("Pruned Drive backup %s (older than %dd)", f["name"], DRIVE_RETENTION_DAYS)
        except Exception as e:
            log.warning("Failed to prune Drive %s: %s", f["name"], e)
    return len(files)


def prune_local_exports() -> int:
    cutoff = datetime.now() - timedelta(days=LOCAL_RETENTION_DAYS)
    removed = 0
    for p in EXPORTS_DIR.glob(f"{VAULT}_*.zip"):
        if datetime.fromtimestamp(p.stat().st_mtime) < cutoff:
            try:
                p.unlink()
                removed += 1
            except Exception as e:
                log.warning("Failed to remove local %s: %s", p, e)
    if removed:
        log.info("Pruned %d local exports older than %dd", removed, LOCAL_RETENTION_DAYS)
    return removed


def main() -> int:
    try:
        local_path = export_vault_local()
        service = get_drive_service()
        folder_id = ensure_drive_folder(service)
        upload_to_drive(service, folder_id, local_path)
        upload_oauth_snapshot(service, folder_id)
        prune_drive_backups(service, folder_id)
        prune_local_exports()
        log.info("Daily backup complete.")
        return 0
    except Exception:
        log.exception("Backup failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
