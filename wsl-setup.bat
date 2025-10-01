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
    echo  DOWNLOADING AI MODEL...
    echo ======================================================================
    echo.
    echo Downloading the AI model file (1.9GB - this may take several minutes)...
    echo.
    
    REM Download the model file using WSL
    set "MODEL_URL=https://huggingface.co/jinggu/jing-model/resolve/main/vit_g_ps14_ak_ft_ckpt_7_clean.pth"
    set "MODEL_PATH=deepcheat/vit_g_ps14_ak_ft_ckpt_7_clean.pth"
    
    echo Downloading from: %MODEL_URL%
    echo Saving to: %MODEL_PATH%
    echo.
    
    REM Try downloading with curl first (with follow redirects)
    wsl -e bash -c "cd '%WSL_PATH%' && if command -v curl >/dev/null 2>&1; then echo 'Using curl to download...'; curl -L --progress-bar -o '%MODEL_PATH%' '%MODEL_URL%' && echo 'Model downloaded successfully with curl!' && exit 0; else echo 'curl not found, trying wget...'; fi; if command -v wget >/dev/null 2>&1; then echo 'Using wget to download...'; wget --show-progress -O '%MODEL_PATH%' '%MODEL_URL%' && echo 'Model downloaded successfully with wget!' && exit 0; else echo 'wget not found'; fi; echo 'Download failed' && exit 1"
    
    if %errorlevel% equ 0 (
        echo.
        echo ✅ AI model downloaded successfully!
        echo.
        
        REM Verify the downloaded file size
        echo Verifying download...
        wsl -e bash -c "cd '%WSL_PATH%' && if [ -f '%MODEL_PATH%' ]; then FILE_SIZE=$(du -h '%MODEL_PATH%' | cut -f1); echo 'Model file size: '$FILE_SIZE; FILE_SIZE_BYTES=$(stat -c%%s '%MODEL_PATH%' 2>/dev/null || stat -f%%z '%MODEL_PATH%' 2>/dev/null); if [ $FILE_SIZE_BYTES -gt 1000000000 ]; then echo '✅ Model file appears to be the correct size'; else echo '⚠️ Warning: Model file seems smaller than expected'; fi; else echo '❌ Model file not found after download'; fi"
        
        echo.
        echo ======================================================================
        echo  SETUP COMPLETE!
        echo ======================================================================
        echo.
        echo The Waldo Alpha application has been successfully installed in WSL!
        echo ✅ AI model is ready to use!
        echo.
    ) else (
        echo.
        echo ❌ Model download failed!
        echo.
        echo ======================================================================
        echo  ⚠️ MANUAL DOWNLOAD REQUIRED
        echo ======================================================================
        echo.
        echo The automatic download failed. Please download the model manually:
        echo.
        echo 1. Go to: https://huggingface.co/jinggu/jing-model/blob/main/vit_g_ps14_ak_ft_ckpt_7_clean.pth
        echo 2. Click the download button
        echo 3. Save the file to: %WSL_PATH%\%MODEL_PATH%
        echo.
        echo The file should be approximately 1.9GB in size.
        echo.
        echo Possible reasons for download failure:
        echo - Network connectivity issues
        echo - Insufficient disk space
        echo - Firewall or proxy restrictions
        echo.
        echo ======================================================================
        echo  SETUP COMPLETE (except model download)
        echo ======================================================================
        echo.
        echo The Waldo Alpha application has been successfully installed in WSL!
        echo ⚠️ Remember to download the AI model manually before using the system.
        echo.
    )
    
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