
@echo off

:REM Check if tools has been downloaded, skip introduction
PUSHD tools && POPD || GOTO FIRSTTIME
GOTO START

:FIRSTTIME
cls
echo Welcome to the one-click comfyui tool to get you started!
echo This version supports INTEL, AMD and NVIDIA GPUs.
echo -
echo PROGRAMS USED:
echo "uv" available at https://docs.astral.sh/uv/#installation
echo "git" available at https://git-scm.com/downloads/win
echo You can open the urls using your mouse and CTRL+C to copy.
echo uv and git can self-install by this script by continuing, but its recommended to manually do it.
echo -
echo uv is a manager that utilizes python to let us run the AI generation code.
echo git allows us to download the ComfyUI codebase to run with Python via uv.
echo -
echo When you are ready,
pause

cls
echo Welcome to the one-click installer!
echo This will download ComfyUI to this directory, so whenever you want to delete it to free space up, you can do so by deleting the "tools" directory that will appear later on.
echo If you encounter any bugs or such, let us know on this repository - or in the associated game's discord server.
echo When you are ready to proceed,
pause

:START

echo Updating...
setlocal enabledelayedexpansion

:: Ensure the script runs in the directory of the batch file
cd /D "%~dp0"

:: Check if uv is installed
where uv >nul 2>&1
if %errorlevel% equ 0 (
	echo "uv has been found!"
) else (
	echo "uv is not installed! Installing via powershell. https://astral.sh/uv/install.ps1"
	powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex" >nul 2>&1
	if %errorlevel% neq 0 (
		echo "I think uv did not install..."
		echo "You can check by running `uv` in the terminal."
		echo "If it displays, its installed, otherwise, install uv manually via the website."
	)
	echo "The terminal needs a reset for uv, so when you are ready to,"
	pause
	exit /b 1
)

:: Check for the git command
where git >nul 2>&1
if %errorlevel% neq 0 (
	echo "git is not installed - attempting auto-install with winget"
	winget install --id Git.Git -e --source winget >nul 2>&1
	if %errorlevel% neq 0 (
		echo "I think git did not install..."
		echo "You can check by running `git` in the terminal."
		echo "Follow the instructions at the start to install git from the website."
	)
	echo The terminal needs a refresh
	echo When you are ready to retry,
	pause
	exit /b 1
)

:: Run the installer
echo ----------------------------------------------------------------
echo Running the installer.
uv run --script installer.py

echo If there is a error here, copy the output and ask for help!
pause
