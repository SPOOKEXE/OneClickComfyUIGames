
## One-Click ComfyUI Game Integration

Easy way to integrate ComfyUI into your games!

View your related section below.

## ComfyUI Installation and Running

### Requirements

1. Install `git` @ https://git-scm.com/downloads
2. Install `Python 3.10.*, 3.11.* or 3.12.*` @ https://www.python.org/downloads/release/python-3129/

Install Python in AppData NOT Program Files! Install Python for your user only and not all users.

### Manual Installation of ComfyUI

For other GPUs apart from NVIDIA, see the OTHER GPUs section below.

1. Open a Command Prompt in the directory where you want ComfyUI to be.
2. Run `git clone https://github.com/comfyanonymous/ComfyUI`
3. Run `cd ComfyUI`
4. Run `cd custom_nodes && git clone https://github.com/Comfy-Org/ComfyUI-Manager && git clone https://github.com/john-mnz/ComfyUI-Inspyrenet-Rembg && cd ..`
5. Run `py -m venv venv`
6. Run `venv\Scripts\activate` for Windows, `venv\bin\activate` for Linux
7. Run `pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126`
8. Run `pip install -r requirements.txt`
9. Run `pip install -r custom_nodes\ComfyUI-Inspyrenet-Rembg\requirements.txt`

### Other GPUs (AMD/MAC/INTEL)

#### For AMD GPUs:
Follow the instructions provided at https://github.com/patientx/ComfyUI-Zluda?tab=readme-ov-file#setup-windows-only for the normal installation.
Note this is a EDITED version of ComfyUI specifically for AMD.

#### For INTEL GPUs
Follow the additional steps provided at https://github.com/comfyanonymous/ComfyUI?tab=readme-ov-file#intel-gpus and https://github.com/comfyanonymous/ComfyUI/discussions/476#discussion-5070898

#### Mac ComfyUI
Follow the instructions provided at https://github.com/comfyanonymous/ComfyUI?tab=readme-ov-file#apple-mac-silicon

### Running

1. Open a Command Prompt in the ComfyUI directory
2. Run `venv\Scripts\activate` for Windows, `venv\bin\activate` for Linux.
3. Run `py main.py --enable-cors-header *`
   1. You can add `--cpu` on the end to run in CPU only
   2. You can add `--lowvram` if you have low VRAM <6GB to help.

For Windows, i highly recommend creating a `start.bat` file in the ComfyUI directory with `venv\Scripts\activate && py main.py --enable-cors-header *` to run ComfyUI easier.
For Linux, the same but `start.sh` and `venv\bin\activate`.

## Developer Game Integration

This is a guide on how to integrate this into your game.

All the code you need is in the `/dev/` folder, specifically the `comfyui.js` file which will let you connect to any ComfyUI instance with `--enable-cores-header *` flag enabled.

You can additionally test this locally using the `comfyui.html` in the `/dev/` folder, which provides a simple interface to generate images.
