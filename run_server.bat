@echo off
setlocal

set "ROOT_DIR=%~dp0"
pushd "%ROOT_DIR%"

echo Starting RealEstate_V2 server on http://127.0.0.1:8000
py -3.11 -m uvicorn api.app:app --host 0.0.0.0 --port 8000

if errorlevel 1 (
    echo.
    echo Server stopped with an error.
    pause
)

popd
endlocal
