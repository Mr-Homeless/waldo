@echo off
setlocal EnableDelayedExpansion

cls
echo ======================================================================
echo  Waldo Alpha - WSL Application Setup
echo ======================================================================
echo.
echo This script will install the Waldo Alpha application inside WSL Ubuntu.
echo It will run the Linux install.sh script within the WSL environment.
echo.
echo Prerequisites:
echo - WSL with Ubuntu must be installed (run wsl-install.bat first)
echo - Ubuntu username and password must be configured
echo.
echo ======================================================================
echo.
pause

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

REM Check if Ubuntu is installed by trying to run a simple command
echo Checking for Ubuntu distribution...
echo Testing direct connection to Ubuntu...
wsl -d Ubuntu echo "Ubuntu test successful" >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Cannot connect to Ubuntu in WSL.
    echo Available distributions:
    wsl --list --verbose
    echo.
    echo Ubuntu appears to be installed but not accessible.
    echo This might be because:
    echo 1. Ubuntu needs to be started for the first time
    echo 2. User account setup is incomplete
    echo.
    echo Try running 'wsl -d Ubuntu' manually to complete setup.
    echo.
    pause
    exit /b 1
)

echo Ubuntu is accessible and working!

echo.
echo STEP 1: Checking WSL Ubuntu status...
echo ----------------------------------------------------------------------

REM Give Ubuntu a moment to fully register (especially right after installation)
timeout /t 3 /nobreak >nul 2>&1

REM Test if we can run commands in WSL using the default distribution
echo Testing WSL connection...
wsl echo "WSL Ubuntu is working!" >nul 2>&1
if %errorlevel% neq 0 (
    echo Cannot connect to default WSL distribution. Trying Ubuntu specifically...
    wsl -d Ubuntu echo "Ubuntu is working!" >nul 2>&1
    if %errorlevel% neq 0 (
        echo ERROR: Cannot connect to WSL Ubuntu.
        echo.
        echo Current WSL distributions:
        wsl --list --verbose
        echo.
        echo Please ensure:
        echo 1. Ubuntu installation completed successfully
        echo 2. You have set up your Ubuntu username and password
        echo 3. WSL service is running
        echo.
        echo Try running: wsl
        echo If Ubuntu doesn't start, you may need to set it up first.
        echo.
        pause
        exit /b 1
    )
)

echo WSL Ubuntu is ready.

echo.
echo STEP 2: Converting Windows path to WSL path...
echo ----------------------------------------------------------------------

REM Get the current directory in Windows format
set "WIN_PATH=%CD%"
echo Windows path: %WIN_PATH%

REM Convert Windows path to WSL path
REM Replace backslashes with forward slashes and C: with /mnt/c
set "WSL_PATH=%WIN_PATH:\=/%"
set "WSL_PATH=%WSL_PATH:C:=/mnt/c%"
set "WSL_PATH=%WSL_PATH:D:=/mnt/d%"
set "WSL_PATH=%WSL_PATH:E:=/mnt/e%"
set "WSL_PATH=%WSL_PATH:F:=/mnt/f%"

echo WSL path: %WSL_PATH%

echo.
echo STEP 3: Checking if install.sh exists...
echo ----------------------------------------------------------------------

wsl test -f "%WSL_PATH%/install.sh"
if %errorlevel% neq 0 (
    echo ERROR: install.sh not found in the current directory.
    echo.
    echo Please ensure you're running this script from the Waldo Alpha
    echo project directory that contains install.sh
    echo.
    pause
    exit /b 1
)

echo install.sh found.

echo.
echo STEP 4: Making install.sh executable...
echo ----------------------------------------------------------------------

wsl chmod +x "%WSL_PATH%/install.sh"
if %errorlevel% neq 0 (
    echo WARNING: Could not set execute permission on install.sh
    echo Attempting to continue anyway...
)

echo.
echo STEP 5: Installing required packages in Ubuntu...
echo ----------------------------------------------------------------------
echo This step ensures Ubuntu has necessary tools like curl...
echo.

REM Update package list and install essential tools
echo Updating Ubuntu package list...
wsl -e bash -c "sudo apt-get update -y"

echo.
echo Installing essential packages...
wsl -e bash -c "sudo apt-get install -y curl wget git build-essential"

echo.
echo STEP 6: Running install.sh in WSL...
echo ----------------------------------------------------------------------
echo This will install Miniconda and set up the Python environment.
echo This process may take 15-30 minutes depending on your internet speed.
echo.
echo Starting installation...
echo ======================================================================
echo.

REM Change to the project directory and run install.sh
wsl -e bash -c "cd '%WSL_PATH%' && ./install.sh"

set INSTALL_RESULT=%errorlevel%

if %INSTALL_RESULT% equ 0 (
    echo.
    echo ======================================================================
    echo  SETUP COMPLETE!
    echo ======================================================================
    echo.
    echo The Waldo Alpha application has been successfully installed in WSL!
    echo.
    echo To run the application:
    echo   1. Use wsl-run.bat (recommended)
    echo   2. Or manually: wsl -e bash -c "cd '%WSL_PATH%' && ./run.sh"
    echo.
    echo The web interface will be available at: http://localhost:5000
    echo.
) else (
    echo.
    echo ======================================================================
    echo  SETUP ENCOUNTERED ISSUES
    echo ======================================================================
    echo.
    echo The installation script reported an error.
    echo.
    echo Common issues and solutions:
    echo 1. Network issues - Check your internet connection and try again
    echo 2. Permission issues - Make sure you set up your Ubuntu user properly
    echo 3. Missing dependencies - Try running: wsl sudo apt-get update
    echo.
    echo You can also try running the installation manually:
    echo   1. Open WSL by typing: wsl
    echo   2. Navigate to: cd "%WSL_PATH%"
    echo   3. Run: ./install.sh
    echo.
)

pause