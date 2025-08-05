@echo off
setlocal enabledelayedexpansion

set "GIT_URL=https://github.com/git-for-windows/git/releases/download/v2.47.1.windows.1/Git-2.47.1-64-bit.exe"
set "PYTHON_URL=https://www.python.org/ftp/python/3.11.8/python-3.11.8-amd64.exe"
set "TEMP_DIR=%TEMP%\installers"

mkdir "%TEMP_DIR%" >nul 2>&1

git --version >nul 2>&1
if %errorlevel% neq 0 (
	echo Git is not installed. Downloading and installing...
	curl -L -o "%TEMP_DIR%\git-installer.exe" !GIT_URL!
	"%TEMP_DIR%\git-installer.exe" /VERYSILENT /NORESTART
) else (
	echo Git is already installed.
)

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

if "%PYTHON_CMD%"=="" (
	echo Python is not installed. Downloading and installing...
	curl -L -o "%TEMP_DIR%\python-installer.exe" !PYTHON_URL!
	"%TEMP_DIR%\python-installer.exe" /quiet InstallAllUsers=0 PrependPath=1
) else (
	echo Python is already installed.
)

rd /s /q "%TEMP_DIR%"
echo Installation complete.
