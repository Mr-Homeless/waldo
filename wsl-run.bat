@echo off
setlocal EnableDelayedExpansion

cls
echo ======================================================================
echo  Waldo Alpha - Starting Application in WSL
echo ======================================================================
echo.

REM Check if any WSL distributions are installed
echo Checking for WSL distributions...
wsl --list --verbose 2>nul | findstr /v "NAME\|----\|Windows Subsystem" | findstr /r "." >nul
if %errorlevel% neq 0 (
    echo ERROR: WSL is not installed or no distributions found.
    echo Please run wsl-install.bat first.
    echo.
    pause
    exit /b 1
)

REM Check if Ubuntu is working
echo Testing Ubuntu connection...
wsl -d Ubuntu echo "SUCCESS: Ubuntu is working"
if errorlevel 1 goto :UBUNTU_ERROR

echo.
echo Ubuntu connection successful!
echo.

echo STEP 2: Converting Windows path to WSL path...
echo ----------------------------------------------------------------------
REM Get the current directory in Windows format
set "WIN_PATH=%CD%"
echo Windows path: %WIN_PATH%

REM Convert Windows path to WSL path
set "WSL_PATH=%WIN_PATH:\=/%"
set "WSL_PATH=%WSL_PATH:C:=/mnt/c%"
set "WSL_PATH=%WSL_PATH:D:=/mnt/d%"
set "WSL_PATH=%WSL_PATH:E:=/mnt/e%"
set "WSL_PATH=%WSL_PATH:F:=/mnt/f%"

echo WSL path: %WSL_PATH%
echo.

echo STEP 3: Checking if run.sh exists...
echo ----------------------------------------------------------------------
wsl -d Ubuntu test -f "%WSL_PATH%/run.sh"
if %errorlevel% neq 0 (
    echo ERROR: run.sh not found in the current directory.
    echo Windows path: %WIN_PATH%
    echo WSL path: %WSL_PATH%
    echo.
    echo Please ensure you're running this script from the Waldo Alpha
    echo project directory that contains run.sh
    echo.
    echo Files in current directory:
    wsl -d Ubuntu ls -la "%WSL_PATH%/" 2>nul || echo "Could not list directory contents"
    echo.
    pause
    exit /b 1
)

echo run.sh found!

echo.
echo STEP 4: Checking conda environment...
echo ----------------------------------------------------------------------
echo Initializing conda...
wsl -d Ubuntu bash -c "source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null; conda env list | grep cs2-detect-env"
if errorlevel 1 goto :CONDA_ERROR

echo.
echo SUCCESS: cs2-detect-env environment found!
echo.
echo ======================================================================
echo  Starting CS2 Cheat Detection System
echo ======================================================================
echo.
echo The web interface will be available at: http://localhost:5000
echo.
echo Press Ctrl+C in this window to stop the server.
echo.
echo Starting server...
echo ----------------------------------------------------------------------
echo.

REM Make run.sh executable (in case it's not)
echo Making run.sh executable...
wsl -d Ubuntu chmod +x "%WSL_PATH%/run.sh"

echo.
echo STEP 5: Starting the CS2 Cheat Detection System...
echo ----------------------------------------------------------------------
echo The web interface will be available at: http://localhost:5000
echo Press Ctrl+C in this window to stop the server.
echo.

REM Run the application with proper conda initialization
echo Launching application with conda environment...
wsl -d Ubuntu bash -c "cd '%WSL_PATH%' && source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null; ./run.sh"

if %errorlevel% neq 0 (
    echo.
    echo ======================================================================
    echo  Application stopped or encountered an error
    echo ======================================================================
    echo.
    echo If you see errors above, common solutions include:
    echo.
    echo 1. Port conflict - Another application may be using port 5000
    echo    Solution: Close other applications or change the port in main.py
    echo.
    echo 2. Missing model files - The .pth model files may not be present
    echo    Solution: Place model files in deepcheat/VideoMAEv2/output/
    echo.
    echo 3. Environment issues - The conda environment may need updating
    echo    Solution: Run wsl-setup.bat again with option to update
    echo.
    echo You can also run manually in WSL:
    echo   1. Open WSL: wsl
    echo   2. Navigate: cd "%WSL_PATH%"
    echo   3. Run: ./run.sh
    echo.
) else (
    echo.
    echo Server stopped successfully.
    echo.
)

pause
exit /b 0

:UBUNTU_ERROR
echo.
echo ======================================================================
echo ERROR: Cannot connect to Ubuntu in WSL
echo ======================================================================
echo.
echo Available distributions:
wsl --list --verbose
echo.
echo Please ensure:
echo 1. Ubuntu is properly installed (run wsl-install.bat if needed)
echo 2. Ubuntu setup is complete (username/password configured)
echo 3. Try running 'wsl -d Ubuntu' manually to test
echo.
pause
exit /b 1

:CONDA_ERROR
echo.
echo ======================================================================
echo ERROR: cs2-detect-env conda environment not found
echo ======================================================================
echo.
echo Debugging conda installation...
echo Checking for miniconda:
wsl -d Ubuntu test -d ~/miniconda3 && echo "miniconda3 directory exists" || echo "miniconda3 directory not found"
echo Checking for anaconda:
wsl -d Ubuntu test -d ~/anaconda3 && echo "anaconda3 directory exists" || echo "anaconda3 directory not found"
echo.
echo Please run wsl-setup.bat first to install the application.
echo.
echo Available conda environments:
wsl -d Ubuntu bash -c "source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null; conda env list 2>/dev/null || echo 'Conda not properly initialized'"
echo.
pause
exit /b 1