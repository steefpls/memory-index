@echo off
REM Daily memory-index work-vault backup. Wrapped by Windows Task Scheduler;
REM also safe to invoke manually for debugging.
REM
REM --no-default-groups: the default `cpu` group would install the CPU
REM onnxruntime over the server's onnxruntime-gpu (same onnxruntime/ folder).
REM The backup never embeds, so it asks for neither; `uv run` syncs inexactly,
REM so whatever ONNX Runtime the venv has is left alone. Without it, the 03:00
REM run on 2026-09-23 installed the CPU build, the 07:22 deploy's exact sync
REM removed it again -- taking the shared files with it -- and search was down
REM until onnxruntime-gpu was reinstalled.
"C:\Users\steve\.local\bin\uv.exe" --directory "C:\Users\steve\Documents\memory-index" run --no-default-groups --extra backup python scripts/backup_to_drive.py
exit /b %ERRORLEVEL%
