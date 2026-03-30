@echo off
setlocal EnableExtensions

if /I "%~1"=="backend" goto backend
if /I "%~1"=="frontend" goto frontend

set "ROOT_DIR=%~dp0"
if "%ROOT_DIR:~-1%"=="\" set "ROOT_DIR=%ROOT_DIR:~0,-1%"

set "BACKEND_DIR=%ROOT_DIR%\backend"
set "FRONTEND_DIR=%ROOT_DIR%\frontend-next"

if not exist "%BACKEND_DIR%\src\api\main.py" (
    echo Backend entrypoint not found: "%BACKEND_DIR%\src\api\main.py"
    pause
    exit /b 1
)

if not exist "%FRONTEND_DIR%\package.json" (
    echo Frontend package.json not found: "%FRONTEND_DIR%\package.json"
    pause
    exit /b 1
)

call :resolve_python
if errorlevel 1 (
    pause
    exit /b 1
)

call :check_npm
if errorlevel 1 (
    pause
    exit /b 1
)

echo Starting backend and frontend-next in separate windows...
start "DataMimicAI Backend" cmd /k ""%~f0" backend"
start "DataMimicAI Frontend Next" cmd /k ""%~f0" frontend"
exit /b 0

:backend
set "ROOT_DIR=%~dp0"
if "%ROOT_DIR:~-1%"=="\" set "ROOT_DIR=%ROOT_DIR:~0,-1%"
set "BACKEND_DIR=%ROOT_DIR%\backend"

cd /d "%BACKEND_DIR%"
call :resolve_python
if errorlevel 1 exit /b 1

echo Starting FastAPI backend on http://localhost:8000
call %BACKEND_PYTHON_CMD% -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
set "BACKEND_EXIT_CODE=%ERRORLEVEL%"
if not "%BACKEND_EXIT_CODE%"=="0" (
    echo.
    echo Backend failed to start.
    echo Install backend dependencies with:
    echo   cd /d "%BACKEND_DIR%"
    echo   %BACKEND_PYTHON_CMD% -m pip install -r requirements.txt
)
exit /b %BACKEND_EXIT_CODE%

:frontend
set "ROOT_DIR=%~dp0"
if "%ROOT_DIR:~-1%"=="\" set "ROOT_DIR=%ROOT_DIR:~0,-1%"
set "FRONTEND_DIR=%ROOT_DIR%\frontend-next"

cd /d "%FRONTEND_DIR%"
call :check_npm
if errorlevel 1 exit /b 1

set "NEXT_PUBLIC_API_URL=http://localhost:8000"

if not exist "%FRONTEND_DIR%\node_modules" (
    echo node_modules not found. Running npm install first...
    call npm install
    if errorlevel 1 (
        echo.
        echo npm install failed in "%FRONTEND_DIR%".
        exit /b 1
    )
)

echo Starting Next.js frontend on http://localhost:3000
call npm run dev
set "FRONTEND_EXIT_CODE=%ERRORLEVEL%"
if not "%FRONTEND_EXIT_CODE%"=="0" (
    echo.
    echo Frontend failed to start.
    echo Install frontend dependencies with:
    echo   cd /d "%FRONTEND_DIR%"
    echo   npm install
)
exit /b %FRONTEND_EXIT_CODE%

:resolve_python
set "ROOT_DIR=%~dp0"
if "%ROOT_DIR:~-1%"=="\" set "ROOT_DIR=%ROOT_DIR:~0,-1%"
set "BACKEND_DIR=%ROOT_DIR%\backend"
set "BACKEND_PYTHON_CMD="

if exist "%BACKEND_DIR%\.venv\Scripts\python.exe" (
    set "BACKEND_PYTHON_CMD="%BACKEND_DIR%\.venv\Scripts\python.exe""
    exit /b 0
)

if exist "%BACKEND_DIR%\venv\Scripts\python.exe" (
    set "BACKEND_PYTHON_CMD="%BACKEND_DIR%\venv\Scripts\python.exe""
    exit /b 0
)

py -3 -V >nul 2>&1
if not errorlevel 1 (
    set "BACKEND_PYTHON_CMD=py -3"
    exit /b 0
)

python -V >nul 2>&1
if not errorlevel 1 (
    set "BACKEND_PYTHON_CMD=python"
    exit /b 0
)

echo Python was not found. Install Python or create backend\.venv first.
exit /b 1

:check_npm
where npm >nul 2>&1
if errorlevel 1 (
    echo npm was not found. Install Node.js so npm is available on PATH.
    exit /b 1
)
exit /b 0