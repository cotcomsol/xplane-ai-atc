@echo off
rem X-Plane AI ATC launcher for Windows. Double-click to run, or: run-atc.bat monitor
cd /d "%~dp0"
where uv >nul 2>nul
if errorlevel 1 (
  echo uv is not installed. Install it with this command in PowerShell, then run this again:
  echo   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  pause
  exit /b 1
)
if not exist atc.toml (
  copy atc.example.toml atc.toml >nul
  echo Created atc.toml - edit it to set your X-Plane folder [xplane] path, then run this again.
  notepad atc.toml
  exit /b 0
)
set CMD=%*
if "%CMD%"=="" set CMD=run
uv run atc %CMD%
pause
