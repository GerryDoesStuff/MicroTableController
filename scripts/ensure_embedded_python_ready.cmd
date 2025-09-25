@echo off
setlocal

for %%I in ("%~dp0..") do set "REPO_ROOT=%%~fI"
pushd "%REPO_ROOT%"

set "PYTHON_EXE=%~1"
if not defined PYTHON_EXE set "PYTHON_EXE=python\python.exe"
if not exist "%PYTHON_EXE%" goto :END

set "SITE_PACKAGES=python\Lib\site-packages"
set "SERIAL_INIT=%SITE_PACKAGES%\serial\__init__.py"

if exist "%SERIAL_INIT%" goto :END

echo [quicklaunch] Provisioning embedded Python dependencies...
call :EnsurePip
if defined HAVE_PIP (
    "%PYTHON_EXE%" -m pip install --disable-pip-version-check --no-warn-script-location -r requirements.txt
    if not errorlevel 1 (
        if exist "%SERIAL_INIT%" goto :END
    )
)

call :InstallPySerialWheel
if exist "%SERIAL_INIT%" goto :END

echo [quicklaunch] WARNING: Unable to provision pyserial into embedded interpreter.

:END
popd
endlocal
exit /b 0

:EnsurePip
"%PYTHON_EXE%" -m pip --version >nul 2>&1
if errorlevel 1 (
    "%PYTHON_EXE%" -m ensurepip --default-pip >nul 2>&1
)
"%PYTHON_EXE%" -m pip --version >nul 2>&1
if errorlevel 1 (
    set "HAVE_PIP="
) else (
    set "HAVE_PIP=1"
)
exit /b 0

:InstallPySerialWheel
set "WHEEL=%REPO_ROOT%\scripts\wheels\pyserial-3.5-py2.py3-none-any.whl"
if not exist "%WHEEL%" (
    echo [quicklaunch] Missing %WHEEL%
    exit /b 0
)
"%PYTHON_EXE%" "%REPO_ROOT%\scripts\install_pyserial_wheel.py"
exit /b 0
