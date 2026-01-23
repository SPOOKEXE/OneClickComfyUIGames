
## One-Click ComfyUI Game Integration

Easy way to integrate ComfyUI into your games!

View your related section below.

## ComfyUI Installation and Running

### Requirements

1. Install `git` @ https://git-scm.com/downloads
2. Install `uv` @ https://docs.astral.sh/uv/#installation

### Manual Installation of ComfyUI

For other GPUs apart from NVIDIA, see the OTHER GPUs section below.

1. Open a Command Prompt in the directory where you want ComfyUI to be.
2. Run `git clone https://github.com/comfyanonymous/ComfyUI`
3. Run `cd ComfyUI`
4. Run `cd custom_nodes && git clone https://github.com/Comfy-Org/ComfyUI-Manager && git clone https://github.com/john-mnz/ComfyUI-Inspyrenet-Rembg && cd ..`
5. Run `uv init && uv venv && uv add pip`
6. [GO TO GPU INSTALLATION SECTION THEN COME BACK]
7. Run `uv pip install -r requirements.txt`
8. Run `uv pip install -r custom_nodes\ComfyUI-Inspyrenet-Rembg\requirements.txt`

### GPU Installation

#### WINDOWS & LINUX

CPU:
`uv pip install torch torchvision torchaudio`

CUDA:
`pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu130`

AMD:
`pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/rocm6.4`

INTEL:
`pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/xpu`

#### MAC

`uv pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu`

### Running

1. Open a Command Prompt in the ComfyUI directory
2. Run `uv run main.py --enable-cors-header *`
   1. You can add `--cpu` on the end to run in CPU only
   2. You can add `--lowvram` if you have low VRAM <6GB to help.

For Windows, i highly recommend creating a `start.bat` file in the ComfyUI directory with `uv run main.py --enable-cors-header *` to run ComfyUI easier.
For Linux, the same but `start.sh`.

## Developer Game Integration

This is a guide on how to integrate this into your game.

All the code you need is in the `/dev/` folder, specifically the `comfyui.js` file which will let you connect to any ComfyUI instance with `--enable-cores-header *` flag enabled.

You can additionally test this locally using the `comfyui.html` which provides a simple interface to generate images.

If you want the one-click file, that is also available in the `/comfyui/` folder.
