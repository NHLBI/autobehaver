@echo off
cd /d "%~dp0.."
python -m autobehaver.desktop %*
if errorlevel 1 pause
