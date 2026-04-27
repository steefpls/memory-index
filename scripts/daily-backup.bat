@echo off
REM Daily memory-index work-vault backup. Wrapped by Windows Task Scheduler;
REM also safe to invoke manually for debugging.
"C:\Users\steve\.local\bin\uv.exe" --directory "C:\Users\steve\Documents\memory-index" run python scripts/backup_to_drive.py
exit /b %ERRORLEVEL%
