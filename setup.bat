@echo off
setlocal

REM Always run from this script's directory so relative paths are portable.
pushd "%~dp0" >nul
set "EXIT_CODE=0"

echo ============================================
echo   Memory Index - Setup (CPU-only)
echo ============================================
echo.

REM Resolve Python launcher: prefer python, fall back to py -3.
set "PYTHON_CMD="
python --version >nul 2>&1 && set "PYTHON_CMD=python"
if not defined PYTHON_CMD (
    py -3 --version >nul 2>&1 && set "PYTHON_CMD=py -3"
)
if not defined PYTHON_CMD (
    echo [ERROR] Python not found. Install Python 3.11+ first.
    set "EXIT_CODE=1"
    goto :cleanup_with_pause
)
echo [OK] Python launcher: %PYTHON_CMD%

REM --- Venv and dependencies ---
if not exist ".venv\Scripts\python.exe" (
    echo [INFO] Creating virtual environment...
    %PYTHON_CMD% -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        set "EXIT_CODE=1"
        goto :cleanup_with_pause
    )
)
set "VENV_PYTHON=%CD%\.venv\Scripts\python.exe"
if not exist "%VENV_PYTHON%" (
    echo [ERROR] Virtual environment Python not found at "%VENV_PYTHON%".
    set "EXIT_CODE=1"
    goto :cleanup_with_pause
)
echo [INFO] Installing Python dependencies...

REM Keep pip current
"%VENV_PYTHON%" -m pip install -q --upgrade pip
if errorlevel 1 (
    echo [WARN] Could not upgrade pip - continuing anyway.
)

echo [INFO] Installing core packages...
"%VENV_PYTHON%" -m pip install -q "mcp[cli]" chromadb networkx onnxruntime
if errorlevel 1 (
    echo [ERROR] Failed to install core packages.
    set "EXIT_CODE=1"
    goto :cleanup_with_pause
)

echo [INFO] Installing embedding model packages...
"%VENV_PYTHON%" -m pip install -q sentence-transformers transformers einops onnxscript
if errorlevel 1 (
    echo [ERROR] Failed to install embedding model packages.
    set "EXIT_CODE=1"
    goto :cleanup_with_pause
)

REM Install project in editable mode
"%VENV_PYTHON%" -m pip install -q -e .
if errorlevel 1 (
    echo [ERROR] Failed to install project in editable mode.
    set "EXIT_CODE=1"
    goto :cleanup_with_pause
)

echo [OK] Dependencies installed

REM --- Download model and export to ONNX ---
echo.
echo [INFO] Downloading CodeRankEmbed model and exporting to ONNX...
echo        (first run downloads ~274MB model, then exports for CPU inference)
echo.

set PYTHONIOENCODING=utf-8
"%VENV_PYTHON%" scripts\export_onnx.py
if errorlevel 1 (
    echo [ERROR] ONNX export failed. The server will still work using PyTorch CPU ^(slower^).
)

REM --- Register MCP server with available AI CLIs ---
echo.
set "REGISTERED_ANY=0"

REM Check for Claude Code
where claude >nul 2>&1
if not errorlevel 1 (
    echo [INFO] Registering MCP server with Claude Code...
    call claude mcp add --scope user memory-index -- "%VENV_PYTHON%" "%CD%\src\server.py" 2>nul
    echo [OK] Registered with Claude Code
    set "REGISTERED_ANY=1"
) else (
    echo [SKIP] Claude Code not found - skipping registration
)

REM Check for Codex CLI
where codex >nul 2>&1
if not errorlevel 1 (
    echo [INFO] Registering MCP server with Codex CLI...
    call codex mcp add memory-index -- "%VENV_PYTHON%" "%CD%\src\server.py" >nul 2>&1
    if errorlevel 1 (
        echo [INFO] codex mcp add not available, writing config directly...
        if not exist "%USERPROFILE%\.codex" mkdir "%USERPROFILE%\.codex"
        if not exist "%USERPROFILE%\.codex\config.toml" type nul > "%USERPROFILE%\.codex\config.toml"

        findstr /C:"[mcp_servers.memory-index]" "%USERPROFILE%\.codex\config.toml" >nul 2>&1
        if not errorlevel 1 (
            echo [INFO] Existing [mcp_servers.memory-index] entry found - skipping direct write.
        ) else (
            set "PYTHON_PATH=%VENV_PYTHON%"
            set "SERVER_PATH=%CD%\src\server.py"
            setlocal enabledelayedexpansion
            set "PYTHON_TOML=!PYTHON_PATH:\=\\!"
            set "SERVER_TOML=!SERVER_PATH:\=\\!"
            >> "%USERPROFILE%\.codex\config.toml" (
                echo.
                echo [mcp_servers.memory-index]
                echo command = "!PYTHON_TOML!"
                echo args = ["!SERVER_TOML!"]
            )
            endlocal
        )
    )
    echo [OK] Registered with Codex CLI
    set "REGISTERED_ANY=1"
) else (
    echo [SKIP] Codex CLI not found - skipping registration
)

if "%REGISTERED_ANY%"=="0" (
    echo.
    echo [WARN] Neither Claude Code nor Codex CLI found.
    echo        You can register the MCP server manually later:
    echo.
    echo        Claude Code:
    echo          claude mcp add --scope user memory-index -- "%VENV_PYTHON%" "%CD%\src\server.py"
    echo.
    echo        Codex CLI:
    echo          codex mcp add memory-index -- "%VENV_PYTHON%" "%CD%\src\server.py"
    pause
)

echo.
echo ============================================
echo   Setup complete!
echo ============================================
echo.
echo Next steps:
echo   1. Restart Claude Code / Codex CLI to load the new MCP server
echo   2. Ask the AI to run: list_vaults()
echo   3. Create a vault: create_vault('default')
echo   4. Add entities: create_entity('Python', 'technology', 'default', 'General purpose language')
echo   5. Search: search_memory('what language')
echo.
goto :cleanup_with_pause

:cleanup_with_pause
pause
goto :cleanup

:cleanup
popd >nul
exit /b %EXIT_CODE%
