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
REM uv lives under the owner's profile: INSTANCE_PROFILE from the environment,
REM else the fleet root's instance.env (INSTANCE_ROOT, default C:\daemon-hub: a
REM client box has one, steef-server doesn't), else steef-server's
REM C:\Users\steve. The checkout is the one this script sits in. The vault,
REM Google account and token path come from the environment: see the
REM backup_to_drive.py docstring.
if not defined INSTANCE_ROOT set "INSTANCE_ROOT=C:\daemon-hub"
if not defined INSTANCE_PROFILE if exist "%INSTANCE_ROOT%\instance.env" (
  for /f "usebackq eol=# tokens=1,* delims==" %%a in ("%INSTANCE_ROOT%\instance.env") do if /i "%%a"=="INSTANCE_PROFILE" set "INSTANCE_PROFILE=%%b"
)
if not defined INSTANCE_PROFILE set "INSTANCE_PROFILE=C:\Users\steve"
"%INSTANCE_PROFILE%\.local\bin\uv.exe" --directory "%~dp0.." run --no-default-groups --extra backup python scripts/backup_to_drive.py
exit /b %ERRORLEVEL%
