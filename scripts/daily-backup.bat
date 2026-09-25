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
REM
REM uv lives under the owner's profile (INSTANCE_PROFILE, default steef-server's
REM C:\Users\steve); the checkout is the one this script sits in. The vault,
REM Google account and token path come from the environment: see the
REM backup_to_drive.py docstring.
if not defined INSTANCE_PROFILE set "INSTANCE_PROFILE=C:\Users\steve"
"%INSTANCE_PROFILE%\.local\bin\uv.exe" --directory "%~dp0.." run --no-default-groups --extra backup python scripts/backup_to_drive.py
exit /b %ERRORLEVEL%
