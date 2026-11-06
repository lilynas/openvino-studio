@echo off
chcp 65001 >nul
cd /d "%~dp0"
start "" pythonw launch_studio.py
if errorlevel 1 (
    python launch_studio.py
)
