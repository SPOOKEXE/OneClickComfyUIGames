#!/bin/bash

# YOU MAY NEED TO RUN \`chmod +x ./one-click-linux-mac.sh\` to run the file
# Or for Linux Mint and similar, right click, allow running, and run file - or open file.
# We recommend running in terminal as it will show you any errors that occur.

# Check if tools has been downloaded, skip introduction
if [ -d "tools" ]; then
    # Directory exists, skip to START
    :
else
    # FIRSTTIME
    clear
    echo "Welcome to the one-click comfyui tool to get you started!"
    echo "This version supports both AMD and NVIDIA GPUs."
    echo ""
    echo "\"uv\" available at https://docs.astral.sh/uv/#installation"
    echo "\"Git\" available at https://git-scm.com/downloads"
    echo "You can select the urls using your mouse and copy them."
    echo "uv and git can self-install by this script by continuing, but its recommended to manually do it."
    echo ""
    echo "uv is a manager that utilizes python to let us run the AI generation code."
    echo "Git allows us to download the ComfyUI codebase to run with Python via uv."
    echo ""
    echo "When you are ready,"
    read -p "press enter to continue..."

    clear
    echo "Welcome to the one-click installer!"
    echo "This will download ComfyUI to this directory, so whenever you want to delete it to free space up, you can do so by deleting the \"tools\" directory that will appear later on."
    echo "If you encounter any bugs or such, let us know on this repository - or in the associated game's discord server."
    echo "When you are ready to proceed,"
    read -p "press enter to continue..."
fi

# START
echo "Updating..."

# Ensure the script runs in the directory of the bash file
cd "$(dirname "$0")"

# Check if uv is installed
if command -v uv &> /dev/null; then
    echo "uv has been found!"
else
    echo "uv is not installed!"
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    echo "The terminal needs a reset for uv, so when you are ready to,"
    read -p "press enter to close the terminal."
    exit 1
fi

# Check for the git command
if ! command -v git &> /dev/null; then
    echo "Git is not installed!"
    echo "Installing git with `sudo apt-get install git`"
    sudo apt-get install git
    echo "To continue, relaunch the bash file,"
    read -p "press enter to close the terminal."
    exit 1
fi

# Run the installer
echo "----------------------------------------------------------------"
echo "Running the installer."
uv run --script installer.py

echo "If there is an error here, copy the output and ask for help!"
read -p "Press enter to exit..."
