@echo off
setlocal EnableDelayedExpansion

cls
echo ======================================================================
echo  Waldo Alpha - WSL Installation Script
echo ======================================================================
echo.
echo This script will install Windows Subsystem for Linux (WSL) with Ubuntu
echo and prepare your system to run the Waldo Alpha application.
echo.
echo NOTE: This process requires administrator privileges and will need
echo a system restart after WSL installation.
echo.
echo ======================================================================
echo.
pause

REM Check for administrator privileges
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo ERROR: This script requires administrator privileges.
    echo.
    echo Please right-click on this file and select "Run as administrator"
    echo or use one of these methods:
    echo   1. Right-click Start Menu, select "Windows Terminal (Admin)"
    echo   2. Type: cd "%~dp0"
    echo   3. Type: wsl-install.bat
    echo.
    pause
    exit /b 1
)

echo.
echo STEP 1: Checking Windows version...
echo ----------------------------------------------------------------------
for /f "tokens=4-5 delims=. " %%i in ('ver') do set VERSION=%%i.%%j
echo Windows version detected: %VERSION%

REM Check if Windows version supports WSL
if "%VERSION%" LSS "10.0" (
    echo ERROR: WSL requires Windows 10 or later.
    echo Your version: %VERSION%
    pause
    exit /b 1
)

echo Windows version is compatible with WSL.
echo.

echo STEP 2: Checking WSL and Ubuntu installation status...
echo ----------------------------------------------------------------------

REM Check if any distributions are installed
wsl --list --quiet >nul 2>&1
if %errorlevel% neq 0 (
    echo WSL is not installed or no distributions found.
    echo Proceeding with full WSL installation...
    goto :INSTALL_WSL
)

REM If we get here, WSL has distributions - check for Ubuntu specifically
echo WSL is installed. Checking for Ubuntu...
wsl --list --quiet | findstr /i "Ubuntu" >nul 2>&1
if %errorlevel% equ 0 (
    echo Ubuntu is already installed.
    echo.
    echo Testing Ubuntu connection...
    wsl -d Ubuntu echo "Test successful" >nul 2>&1
    if %errorlevel% equ 0 (
        echo Ubuntu is working properly.
        echo You can proceed directly to wsl-setup.bat
        echo.
        pause
        exit /b 0
    ) else (
        echo Ubuntu is installed but not working properly.
        echo Reinstalling Ubuntu...
        goto :INSTALL_UBUNTU
    )
) else (
    echo Ubuntu is not installed. Installing Ubuntu...
    goto :INSTALL_UBUNTU
)

:INSTALL_WSL
echo.
echo STEP 3: Enabling required Windows features...
echo ----------------------------------------------------------------------
echo This may take several minutes...
echo.

REM Enable Windows Subsystem for Linux
echo Enabling Windows Subsystem for Linux...
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
if %errorlevel% neq 0 (
    echo ERROR: Failed to enable WSL feature.
    pause
    exit /b 1
)

REM Enable Virtual Machine Platform for WSL 2
echo.
echo Enabling Virtual Machine Platform...
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
if %errorlevel% neq 0 (
    echo WARNING: Could not enable Virtual Machine Platform.
    echo WSL 1 will be used instead of WSL 2.
)

echo.
echo STEP 4: Installing WSL and Ubuntu...
echo ----------------------------------------------------------------------
echo This will download and install Ubuntu. This may take 10-20 minutes...
echo.

REM Install WSL with Ubuntu
wsl --install -d Ubuntu
if %errorlevel% neq 0 (
    echo.
    echo WARNING: WSL installation may have encountered issues.
    echo Attempting alternative installation method...
    echo.

    REM Try installing just WSL first
    wsl --install --no-distribution

    REM Then install Ubuntu separately
    :INSTALL_UBUNTU
    echo.
    echo Installing Ubuntu distribution...
    wsl --install -d Ubuntu

    if %errorlevel% neq 0 (
        echo.
        echo ERROR: Failed to install Ubuntu.
        echo.
        echo Please try the following:
        echo 1. Restart your computer
        echo 2. Run this script again
        echo 3. If it still fails, install Ubuntu manually from Microsoft Store
        echo.
        pause
        exit /b 1
    )
)

echo.
echo STEP 5: Setting WSL 2 as default version...
echo ----------------------------------------------------------------------
wsl --set-default-version 2 >nul 2>&1
if %errorlevel% equ 0 (
    echo WSL 2 set as default version.
) else (
    echo WSL 1 will be used (WSL 2 not available on this system).
)

echo.
echo ======================================================================
echo  INSTALLATION COMPLETE - RESTART REQUIRED
echo ======================================================================
echo.
echo WSL and Ubuntu have been installed successfully!
echo.
echo IMPORTANT: You must restart your computer before proceeding.
echo.

REM Cancel any pending reboots first
echo Canceling any pending system reboots...
shutdown /a >nul 2>&1

echo After restarting:
echo 1. Ubuntu will prompt you to create a username and password
echo 2. Remember these credentials - you'll need them
echo 3. Run wsl-setup.bat to install the Waldo Alpha application
echo.
echo ======================================================================
echo  IMPORTANT: Windows may show an automatic reboot prompt
echo ======================================================================
echo.
echo If you see a BLUE SCREEN with a countdown timer asking to restart:
echo   - Press ENTER to CANCEL the automatic reboot
echo   - This gives you control over when to restart
echo   - You can then choose when to restart below
echo.
echo ======================================================================
echo.
echo Would you like to restart now? (Y/N)
choice /c YN /m "Restart computer"
if errorlevel 2 (
    echo.
    echo ======================================================================
    echo  Manual Restart Required
    echo ======================================================================
    echo.
    echo Please restart your computer manually when ready.
    echo.
    echo IMPORTANT: If Windows shows a blue restart prompt with countdown:
    echo   1. Press ENTER to cancel the automatic restart
    echo   2. Save your work
    echo   3. Restart manually when ready
    echo.
    echo After restarting, run wsl-setup.bat to continue installation.
    echo.
    pause
    exit /b 0
) else (
    echo.
    echo ======================================================================
    echo  Initiating Restart
    echo ======================================================================
    echo.
    echo Restarting in 10 seconds...
    echo Save any open work now!
    echo.
    echo If a blue screen appears asking about restart:
    echo   - Let the countdown complete OR press ENTER to proceed
    echo.
    timeout /t 10
    shutdown /r /t 0
)