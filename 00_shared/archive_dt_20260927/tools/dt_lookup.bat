@echo off
chcp 65001 >nul
python "%~dp0dt_lookup.py" %*
if errorlevel 1 pause
