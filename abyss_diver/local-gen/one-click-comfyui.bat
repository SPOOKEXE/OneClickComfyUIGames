
@echo off

:REM Check if tools has been downloaded, skip introduction
PUSHD tools && POPD || GOTO FIRSTTIME
GOTO START

:FIRSTTIME
cls
echo Welcome to the one-click comfyui tool to get you started!
echo This version supports both AMD and NVIDIA GPUs on Windows.
echo You will need to install TWO programs before continuing.
echo -
echo "Python 3.11.9" available at https://www.python.org/downloads/release/python-3119/
echo "Git" available at https://git-scm.com/downloads/win
echo You can select the urls using your mouse and CTRL+C to copy.
echo -
echo Python lets us run the AI generation code.
echo Git allows us to download the code to run with Python.
echo -
echo When you have finished installing both of these,
pause

echo -
echo Assuming you have now installed BOTH Git and Python 3.11.9,
echo You can press any key to now launch the install process.
echo When you are ready to go,
pause

cls
echo Welcome to the one-click installer for Windows!
echo This will download ComfyUI to this directory, so whenever you want to delete it to free space up, you can do so by deleting the "tools" directory that will appear later on.
echo If you encounter any bugs or such, let us know on this repository - or in the associated game's discord server.
echo When you are ready to proceed,
pause

:START

echo Updating...
setlocal enabledelayedexpansion

:: Ensure the script runs in the directory of the batch file
cd /D "%~dp0"

:: Check for 'py', 'python', or 'python3' in order and set PYTHON_CMD
set "PYTHON_CMD="
where py >nul 2>&1
if %errorlevel% equ 0 (
	set "PYTHON_CMD=py"
) else (
	where python >nul 2>&1
	if %errorlevel% equ 0 (
		set "PYTHON_CMD=python"
	) else (
		where python3 >nul 2>&1
		if %errorlevel% equ 0 (
			set "PYTHON_CMD=python3"
		)
	)
)

:: If Python command is still not set, install Python
if "%PYTHON_CMD%"=="" (
	echo Python is not installed, installing now silently.
	call install_git_python.bat
	pause
	exit /b 1
)

:: Check for the git command
where git >nul 2>&1
if %errorlevel% neq 0 (
	echo Git is not installed, installing now silently.
	call install_git_python.bat
	pause
	exit /b 1
)

:: Attempt ensurepip
echo Ensure pip (ignore if error)
"%PYTHON_CMD%" -m ensurepip

:: Install required Python packages
echo Upgrading pip.
"%PYTHON_CMD%" -m pip install --upgrade pip
echo Installing installer.py packages.
"%PYTHON_CMD%" -m pip install tqdm requests

:: Run the installer.py
cls
echo Running the installer_windows.py
call "%PYTHON_CMD%" installer_windows.py

echo If there is a error here, copy the output and ask for help!
pause
