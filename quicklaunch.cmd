@echo off
setlocal
set "ROOT=%~dp0"
set "SCRIPT=%ROOT%scripts\quicklaunch.py"

if exist "%ROOT%pythonw.exe" (
    "%ROOT%pythonw.exe" "%SCRIPT%"
) else if exist "%ROOT%python.exe" (
    "%ROOT%python.exe" "%SCRIPT%"
) else (
    python.exe "%SCRIPT%"
)
endlocal
