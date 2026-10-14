@echo off
rem ComfyFront_sep MVP launcher: start the local server (it opens the browser
rem itself after the socket is bound) and keep this window for server logs.
cd /d "%~dp0"
title ComfyFront_sep server
"P:\Software\python-3.14.5-embed-amd64\python.exe" serve.py 8188
pause
