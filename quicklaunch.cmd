@echo off
setlocal
pushd "%~dp0"

set "ADJUSTED_ROOT="
if not exist microstage_app\__init__.py if exist MicroTableController_v3\microstage_app\__init__.py (
    pushd MicroTableController_v3
    set "ADJUSTED_ROOT=1"
)

set "ACTIVATE_SCRIPT="
if exist .venv\Scripts\activate.bat (
    set "ACTIVATE_SCRIPT=.venv\Scripts\activate.bat"
) else if exist microstage_app\.venv\Scripts\activate.bat (
    set "ACTIVATE_SCRIPT=microstage_app\.venv\Scripts\activate.bat"
)

if defined ACTIVATE_SCRIPT (
    call "%ACTIVATE_SCRIPT%"
)

set "PYTHON_EXE="
if exist .venv\Scripts\python.exe (
    set "PYTHON_EXE=.venv\Scripts\python.exe"
) else if exist microstage_app\.venv\Scripts\python.exe (
    set "PYTHON_EXE=microstage_app\.venv\Scripts\python.exe"
) else if exist python\python.exe (
    set "PYTHON_EXE=python\python.exe"
)

if /I "%PYTHON_EXE%"=="python\python.exe" if exist scripts\ensure_embedded_python_ready.cmd (
    call scripts\ensure_embedded_python_ready.cmd "%PYTHON_EXE%"
)

if not defined PYTHON_EXE (
    set "PYTHON_EXE=python"
)

"%PYTHON_EXE%" -m microstage_app

if defined ADJUSTED_ROOT popd
popd
endlocal
pause
