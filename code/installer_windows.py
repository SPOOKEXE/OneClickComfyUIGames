
'''
Windows:
1. git clone ComfyUI
2. python -m venv venv
3. install requirements
4. download custom_nodes
5. install custom_nodes requirements to python_embeded
6. download Checkpoints to models/checkpoints & models/loras
7. run comfyui
'''

'''
Note:
- NVIDIA Cuda uses ComfyUI base
- AMD RomC uses repository 'patientx/ComfyUI-Zluda'
'''

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import os
import platform
import subprocess
import threading
import logging
import time
import re
import json
import sys

def get_fflags() -> dict:
	if not os.path.exists("fflag"):
		return dict()
	with open("fflag", "r", encoding='utf-8', errors='ignore') as file:
		try:
			flags : dict = json.loads(file.read())
			assert isinstance(flags, dict), "FFlags is not a dictionary!"
		except:
			return dict()
	return flags

def get_fflag(key : str) -> bool:
	fflags = get_fflags()
	return fflags.get(key, False)

def set_fflag(key : str, value : bool):
	fflags = get_fflags()
	fflags[key] = value
	with open("fflag", "w", encoding='utf-8', errors='ignore') as file:
		file.write(json.dumps(fflags, indent=4))

INSTALLER_DIRECTORY : Path = Path(os.path.dirname(os.path.abspath(__file__)))
TOOLS_DIRECTORY : Path = INSTALLER_DIRECTORY / "tools"

SUPPORTED_PYTHON_VERSIONS : Tuple[str] = ("3.10", "3.11")

COMFYUI_MAIN_REPOSITORY_URL : str = "https://github.com/comfyanonymous/ComfyUI"
COMFYUI_AMD_GPU_REPOSITORY_URL : str = "https://github.com/patientx/ComfyUI-Zluda"

COMMAND_LINE_ARGS_FOR_COMFYUI : List[str] = [
	# custom command line arguments for comfyui - separate each string by comma
	# e.g. ["--listen", "--api", "--highvram", "--somevalue", "5"]

]

COMFYUI_CUSTOM_NODES : List[str] = [
	"https://github.com/ltdrdata/ComfyUI-Manager",
	"https://github.com/john-mnz/ComfyUI-Inspyrenet-Rembg"
]

AI_CHECKPOINT_DOWNLOADS : Dict[str, Optional[str]] = {
	# SD1.5
	"hassakuHentaiModel_v13.safetensors" : None,
	# PonyXL
	"hassakuXLPony_v13BetterEyesVersion.safetensors" : "https://huggingface.co/FloricSpacer/AbyssDiverModels/resolve/main/hassakuXLPony_v13BetterEyesVersion.safetensors?download=true"
}

AI_LORA_DOWNLOADS : Dict[str, Optional[str]] = {
	# SD1.5
	"midjourneyanime.safetensors" : None,
	# PonyXL
	"DallE3-magik.safetensors" : "https://huggingface.co/FloricSpacer/AbyssDiverModels/resolve/main/DallE3-magik.safetensors?download=true"
}

def run_subprocess_cmd(arguments : List[str], **kwargs) -> Optional[subprocess.CompletedProcess]:
	"""Run a subprocess command with the essential kwargs."""
	try:
		return subprocess.run(arguments, capture_output=True, text=True, check=True, shell=True, **kwargs)
	except Exception as e:
		print(e)
		return None

def run_command(args: List[str] | str, shell: bool = False, cwd : Optional[str] = None, env : os._Environ | dict = os.environ) -> Tuple[int, str]:
	"""Run the following command using subprocess and read the output live to the user. DO NOT USE IF YOU NEED TO PROMPT THE USER."""
	print(f'RUNNING COMMAND: {str(args)}')
	print('=' * 10)
	try:
		process = subprocess.Popen(
			args,
			shell=shell,
			stdout=subprocess.PIPE,
			stderr=subprocess.PIPE,
			text=True,
			bufsize=1,
			cwd=cwd,
			env=env
		)
		stdout_var : str = ""
		def stream_reader(pipe, log_level):
			nonlocal stdout_var
			"""Reads from a pipe and logs each line at the given log level."""
			with pipe:
				for line in iter(pipe.readline, ''):
					print(log_level, line.strip())
					stdout_var += line.strip()
		# Use threads to prevent blocking
		stdout_thread = threading.Thread(target=stream_reader, args=(process.stdout, logging.INFO))
		stderr_thread = threading.Thread(target=stream_reader, args=(process.stderr, logging.ERROR))
		stdout_thread.start()
		stderr_thread.start()
		# Wait for the process and threads to complete
		process.wait()
		stdout_thread.join()
		stderr_thread.join()
		status_code = process.returncode
		if status_code == 0:
			print(f"Command succeeded: {str(args)}")
		else:
			print(f"Command failed with code {status_code}: {str(args)}")
		return status_code, stdout_var
	except Exception as e:
		print(f"Command execution exception: {str(args)}")
		print(f"Exception details: {e}")
		return -1, str(e)

def get_system_python_command() -> Tuple[Path, str]:
	# get the python interpretor running this file
	cmd : str = Path(sys.executable).absolute()
	# check version
	output = run_subprocess_cmd([cmd, "--version"])
	assert output.returncode == 0, f"Python `{cmd} --version` is not available."
	# extract version
	version = re.search(r"Python (\d+\.\d+\.\d+)", output.stdout)
	assert version, "Python command output no version!"
	print(f"Extracted Python Version: {version.group(1)}")
	# check if valid version(s)
	vrs = ", ".join(SUPPORTED_PYTHON_VERSIONS)
	assert version.group(1).startswith(SUPPORTED_PYTHON_VERSIONS), f"Unsupported Python Version! Must be one of {vrs}"
	# return the command
	return cmd

SYSTEM_PYTHON_PATH : Path = get_system_python_command()

print("Checking for missing essential python packages...")
needs_restart : bool = False

# see if requests is installed
try:
	import requests
except:
	print("Package 'requests' is not installed! Installing now.")
	run_subprocess_cmd([SYSTEM_PYTHON_PATH, "-m", "pip", "install", "requests"])
	needs_restart = True

# see if tqdm is installed (progress bars)
try:
	from tqdm import tqdm
except:
	print("Package 'tqdm' is not installed! Installing now.")
	run_subprocess_cmd([SYSTEM_PYTHON_PATH, "-m", "pip", "install", "tqdm"])
	needs_restart = True

if needs_restart:
	print("The script needs to be restarted as new packages were installed.")
	print("Press enter to close the terminal, then re-run the 'one_click_windows.bat'.")
	input("")
	exit()
print("Got all essential python packages!")

def assert_path_length_limit() -> None:
	"""Check how long the path is for the local-gen folder."""
	current_path = Path(os.path.abspath(os.getcwd()))
	path_length : int = len(current_path.as_posix()) + 70 # 70 being approx submodules of ComfyUI
	print(f"Current path: {current_path}")
	print(f"Path length: {path_length} characters")
	if path_length > 260:
		print("Warning: Path length exceeds the Windows path limit of 260 characters. Please move the abyss diver game folder elsewhere.")
		print("Press enter to continue...")
		input("")
		exit()
	if path_length > 240:
		print("Warning: Path length is close to the Windows path limit. Please move the abyss diver game folder elsewhere.")
		print("Press enter to continue...")
		input("")
		exit()
	print("Path length is within safe limits. The installer will continue.")
assert_path_length_limit()

def download_file(url: str, filepath: Path, chunk_size: int = 64) -> None:
	"""Download file from the url to the filepath, chunk_size is how much data is downloaded at once in the stream."""
	response = requests.get(url, stream=True) # type: ignore
	response.raise_for_status()  # Raise an error for bad status codes
	total_size = int(response.headers.get('content-length', 0))
	progress_bar = tqdm(total=total_size, unit='B', unit_scale=True, desc=filepath.as_posix().split('/')[-1]) # type: ignore
	with open(filepath.as_posix(), 'wb') as file:
		for chunk in response.iter_content(chunk_size=chunk_size * 1024):
			file.write(chunk)
			progress_bar.update(len(chunk))
	progress_bar.close()
	print(f"File downloaded to {filepath.as_posix()}")

def check_for_proxy_and_comfyui_responses() -> None:
	"""Ping the proxy on 127.0.0.1:12500/docs and ComfyUI on 127.0.0.1:8188 to see if both are available to the user."""
	import requests

	# wait a period of time as it may take a second to install.
	# TODO: may be too fast for people with slow internet - find a better method
	time.sleep(20)

	proxy_ip : str = "http://127.0.0.1:12500/echo"
	try:
		r = requests.get(proxy_ip)
		if r.status_code != 200:
			raise
	except:
		print(f"Cannot connect to the proxy on {proxy_ip}! The proxy may have not started in time or failed to startup!")

	comfyui_ip = "http://127.0.0.1:8188"
	try:
		r = requests.get(comfyui_ip)
		if r.status_code != 200: raise
	except:
		print(f"Cannot connect to ComfyUI on {comfyui_ip}! ComfyUI may not have started or failed to startup!")

	print("Successfully connected to both ComfyUI and the Proxy!")
	print("Head to Abyss Diver and open the AI Portrait page!")
	print("")

def clone_custom_nodes_to_folder(CUSTOM_NODES_FOLDER : Path) -> None:
	"""Download all the stored comfyui custom nodes to the given folder"""
	previous_directory = Path(os.getcwd()).absolute()
	os.makedirs(CUSTOM_NODES_FOLDER.as_posix(), exist_ok=True)
	os.chdir(CUSTOM_NODES_FOLDER.as_posix())
	for node_repository_url in COMFYUI_CUSTOM_NODES:
		# attempt to clone all repositories into the directory
		print(f'Cloning: {node_repository_url}')
		_ = run_subprocess_cmd(["git", "clone", node_repository_url], cwd=CUSTOM_NODES_FOLDER.as_posix())
	os.chdir(previous_directory)

def download_checkpoints_to_subfolder(models_folder : Path) -> None:
	"""Download the checkpoints to the sub-folder checkpoints"""
	for filename, checkpoint_url in AI_CHECKPOINT_DOWNLOADS.items():
		if checkpoint_url is None:
			print(f"{filename} checkpoint has no download set yet.")
			continue
		checkpoint_filepath : Path = models_folder / filename
		if os.path.exists(checkpoint_filepath) is True:
			print(f"Checkpoint {filename} is already installed.")
			continue
		try:
			download_file(checkpoint_url, checkpoint_filepath)
		except Exception as e:
			print(f'Failed to download checkpoint {filename}.')
			print(e)
			try:
				os.remove(checkpoint_filepath)
			except:
				pass
			assert False, f"Failed to download the {filename} checkpoint file."

def download_loras_to_subfolder(models_folder : Path) -> None:
	"""Download the checkpoints to the sub-folder loras"""
	for filename, lora_url in AI_LORA_DOWNLOADS.items():
		if lora_url is None:
			print(f'{filename} lora has no download set yet.')
			continue
		lora_filepath : Path = models_folder / filename
		if os.path.exists(lora_filepath) is True:
			print(f"Checkpoint {filename} is already installed.")
			continue
		try:
			download_file(lora_url, lora_filepath)
		except Exception as e:
			print(f'Failed to download lora {filename}.')
			print(e)
			try:
				os.remove(lora_filepath)
			except:
				pass
			assert False, f"Failed to download the {filename} lora file."

def check_python_torch_compiled_with_cuda(python_file : Path) -> bool:
	try:
		status, _ = run_command([python_file.as_posix(), "-c" "import torch; assert torch.cuda.is_available(), \'cuda not available\'"], shell=True)
		assert status == 0, "Torch failed to import."
		return True
	except:
		return False

def update_python_torch_compiled_cuda(python_file : Path) -> bool:
	print("="*10)
	print("Installing torch torchaudio and torchvision with CUDA acceleration.")
	print("Please open a new terminal, type 'nvidia-smi' and find the CUDA Version: XX.X.")
	print("If nvidia-smi is not a valid command, please install a NVIDIA graphics driver and restart the terminal.")
	print("You will be asked for either CUDA 11.8, 12.1, or 12.4.")
	print("If you do not have any of these listed versions, select the one closest to yours.")
	print(" ")
	if input("Are you using CUDA 11.8? (y/n)").lower() == "y":
		print("Cuda 11.8 was selected.")
		index_url = "https://download.pytorch.org/whl/cu118"
	elif input("Are you using CUDA 12.1? (y/n)").lower() == "y":
		print("Cuda 12.1 was selected.")
		index_url = "https://download.pytorch.org/whl/cu121"
	elif input("Are you using CUDA 12.4? (y/n)").lower() == "y":
		print("Cuda 12.4 was selected.")
		index_url = "https://download.pytorch.org/whl/cu124"
	else:
		print("Defaulted to CUDA 12.4.")
		index_url = "https://download.pytorch.org/whl/cu124"
	_, __ = run_command([python_file.as_posix(), "-m", "pip", "install", "--upgrade", "torch", "torchaudio", "torchvision", "--index-url", index_url], shell=True)
	print(f"Installed {index_url} cuda acceleration for torch.")

def comfyui_installed_shared_requirements(COMFYUI_DIRECTORY : Path):
	VENV_DIRECTORY = COMFYUI_DIRECTORY / "venv"
	VENV_PYTHON_FILEPATH = VENV_DIRECTORY / "Scripts" / "python.exe"
	print(f"Venv Python Directory: {VENV_PYTHON_FILEPATH.as_posix()}")

	if os.path.exists(VENV_PYTHON_FILEPATH) is False:
		print(f'No virtual environment located - creating now!')
		try:
			completed_process = run_subprocess_cmd([SYSTEM_PYTHON_PATH.as_posix(), "-m", "venv", VENV_DIRECTORY.as_posix()])
			assert completed_process, f"Failed to create the venv directory!"
			status = completed_process.returncode
		except:
			status = None
		print(f"venv create status: {status}")
		assert status == 0, "Failed to create a virtual environment in the ComfyUI folder."

	# try use that python file
	try:
		completed_process = run_subprocess_cmd([VENV_PYTHON_FILEPATH.as_posix(), "--version"])
		assert completed_process, "Failed to run the command."
		status = completed_process.returncode
	except:
		status = None
	print(f'Venv run python version status: {status}')

	print("Venv python.exe does exist." if os.path.exists(VENV_PYTHON_FILEPATH) else "Venv python.exe does NOT exist!")
	assert status == 0, "Failed to activate the virtual environment."

	# install proxy requirements
	if not get_fflag("proxy_requirements_installed"):
		print('Installing proxy.py requirements.')
		packages : List[str] = ["tqdm", "requests", "fastapi", "pydantic", "pillow", "websocket-client", "aiohttp", "uvicorn", "websockets"]
		_, __ = run_command([VENV_PYTHON_FILEPATH.as_posix(), "-m", "pip", "install"] + packages, shell=True)
		set_fflag("proxy_requirements_installed", True)

def comfyui_amd() -> None:
	COMFYUI_DIRECTORY = TOOLS_DIRECTORY / "ComfyUI-Zluda"
	print(f'ComfyUI AMD install directory: {COMFYUI_DIRECTORY.as_posix()}')

	if os.path.exists(COMFYUI_DIRECTORY) is False:
		print("Attempting to clone ComfyUI to the directory.")
		repository_url = COMFYUI_AMD_GPU_REPOSITORY_URL
		previous_directory = os.getcwd()
		os.chdir(TOOLS_DIRECTORY)
		try:
			completed_process = run_subprocess_cmd(["git", "clone", repository_url])
			assert completed_process, "Failed to run the command."
			status = completed_process.returncode
		except Exception as e:
			print(e)
			status = None
		print(f"git clone status: {status}")
		os.chdir(previous_directory)

	assert os.path.exists(COMFYUI_DIRECTORY), f"Failed to clone the ComfyUI repository to {COMFYUI_DIRECTORY.as_posix()}"

	print("Due to how the AMD GPU version needs to be support, you will have to do some manual dependency installation following the repository's guide.")
	if input("Have you installed the dependencies needed already? (y/n) ").lower() == "n":
		print(COMFYUI_AMD_GPU_REPOSITORY_URL + "?tab=readme-ov-file#dependencies")
		print("Open this repository and follow the guide to install the dependencies.")
		print(f"The ComfyUI directory can be found at: {COMFYUI_DIRECTORY.absolute().as_posix()}")
		print("Press enter to restart the one-click...")
		input("")
		exit(1)

	print("Dependencies have been installed.")
	comfyui_installed_shared_requirements(COMFYUI_DIRECTORY)
	VENV_DIRECTORY = COMFYUI_DIRECTORY / "venv"
	VENV_PYTHON_FILEPATH = VENV_DIRECTORY / "Scripts" / "python.exe"

	# start comfyui
	env = dict(
		os.environ,
		ZLUDA_COMGR_LOG_LEVEL="1",
		python=VENV_PYTHON_FILEPATH.as_posix(),
		py=VENV_PYTHON_FILEPATH.as_posix(),
		VENV_DIR=VENV_DIRECTORY.as_posix()
	)

	print("Only certain AMD gpus are actually supported and can be viewed at https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html")
	print("Do you have an older or unsupported AMD card? (y/n)? ")
	if input("Note: this is a experimental workaround and if this fails your device is not supported. ").lower() == "y":
		env['HSA_OVERRIDE_GFX_VERSION'] = "10.3.0"

	if input("Do you have built-in graphics in your CPU (y/n)? ").lower() == "y":
		env['HIP_VISIBLE_DEVICES'] = "1"

	# force all environemnts to be string
	for key, value in env.items():
		if not isinstance(value, str):
			env[key] = str(value)

	zluda_zip = COMFYUI_DIRECTORY / "zluda.zip"
	if os.path.exists(zluda_zip):
		print("Left over zluda.zip was found - deleting.")
		os.remove(zluda_zip)

	ZLUDA_PATCHZLUDA_BATCH = COMFYUI_DIRECTORY / "patchzluda.bat"
	print(f"Running {ZLUDA_PATCHZLUDA_BATCH.as_posix()} with cwd={COMFYUI_DIRECTORY.as_posix()}")
	assert os.path.exists(ZLUDA_PATCHZLUDA_BATCH), "Could not find the patchzluda.bat in the ComfyUI-Zluda directory."
	_, __ = run_command([ZLUDA_PATCHZLUDA_BATCH.as_posix()], cwd=COMFYUI_DIRECTORY.as_posix())

	ZLUDA_INSTALL_BATCH = COMFYUI_DIRECTORY / "install.bat"
	print(f"Running {ZLUDA_INSTALL_BATCH.as_posix()} with cwd={COMFYUI_DIRECTORY.as_posix()}")
	assert os.path.exists(ZLUDA_INSTALL_BATCH), "Could not find the install.bat in the ComfyUI-Zluda directory."

	PROXY_PYTHON_FILE = INSTALLER_DIRECTORY / "proxy.py"
	assert os.path.exists(PROXY_PYTHON_FILE), "The local image generation proxy.py does not exist!"

	command2_args = [VENV_PYTHON_FILEPATH.as_posix(), PROXY_PYTHON_FILE.as_posix()]
	print("Running Proxy with the following commands:")
	print(command2_args)

	print("Starting both ComfyUI and Proxy scripts.")

	thread1 = threading.Thread(target=lambda : run_command([ZLUDA_INSTALL_BATCH.as_posix()], env=env, cwd=COMFYUI_DIRECTORY.as_posix()))
	thread2 = threading.Thread(target=lambda : run_command(command2_args, env=env, cwd=INSTALLER_DIRECTORY.as_posix()))
	# thread3 = threading.Thread(target=lambda : check_for_proxy_and_comfyui_responses())
	thread1.start()
	thread2.start()
	# thread3.start()
	thread1.join()
	thread2.join()
	# thread3.join()

	print("Both ComfyUI and Proxy scripts have finished.")

def comfyui_nvidia() -> None:
	COMFYUI_DIRECTORY = TOOLS_DIRECTORY / "ComfyUI"
	print(f'ComfyUI install directory: {COMFYUI_DIRECTORY.as_posix()}')

	if os.path.exists(COMFYUI_DIRECTORY) is False:
		print("Cloning the ComfyUI repository...")
		previous_directory = Path(os.getcwd()).absolute()
		os.chdir(TOOLS_DIRECTORY)
		try:
			completed_process = run_subprocess_cmd(["git", "clone", COMFYUI_MAIN_REPOSITORY_URL])
			assert completed_process, "Failed to run the command."
			status = completed_process.returncode
		except Exception as e:
			print(e)
			status = None
		print(f"git clone status: {status}")
		os.chdir(previous_directory)

	assert os.path.exists(COMFYUI_DIRECTORY), f"Failed to clone the ComfyUI repository to {COMFYUI_DIRECTORY.as_posix()}"

	comfyui_installed_shared_requirements(COMFYUI_DIRECTORY)
	VENV_DIRECTORY = COMFYUI_DIRECTORY / "venv"
	VENV_PYTHON_FILEPATH = VENV_DIRECTORY / "Scripts" / "python.exe"

	# install ComfyUI/requirements.txt
	if not get_fflag("comfyui_requirements_installed"):
		print('Installing ComfyUI requirements.')
		requirements_file = COMFYUI_DIRECTORY / "requirements.txt"
		_, __ = run_command([VENV_PYTHON_FILEPATH.as_posix(), "-m", "pip", "install", "-r", requirements_file], shell=True)
		set_fflag("comfyui_requirements_installed", True)

	# git clone custom_nodes
	print('Cloning all custom nodes.')
	CUSTOM_NODES_FOLDER = COMFYUI_DIRECTORY / "custom_nodes"
	clone_custom_nodes_to_folder(CUSTOM_NODES_FOLDER)

	# pip install custom_nodes requirements.txt
	print('Installing custom nodes requirements.')
	for folder_name in os.listdir(CUSTOM_NODES_FOLDER):
		if get_fflag(f"node_{folder_name}_requirements_installed"):
			continue
		TARGET_FOLDER_REQUIREMENTS_FILE = CUSTOM_NODES_FOLDER / folder_name / "requirements.txt"
		if os.path.exists(TARGET_FOLDER_REQUIREMENTS_FILE) is False:
			continue # cannot find requirements.txt for this item name
		print(f"Custom Nodes requirements filepath: {TARGET_FOLDER_REQUIREMENTS_FILE.as_posix()}")
		_, __ = run_command([VENV_PYTHON_FILEPATH.as_posix(), "-m", "pip", "install", "-r", TARGET_FOLDER_REQUIREMENTS_FILE.as_posix()], shell=True)
		set_fflag(f"node_{folder_name}_requirements_installed", True)

	# download all checkpoint and lora models
	print("Downloading missing checkpoints...")
	download_checkpoints_to_subfolder(COMFYUI_DIRECTORY / "models" / "checkpoints")

	print("Downloading missing loras...")
	download_loras_to_subfolder(COMFYUI_DIRECTORY / "models" / "loras")

	arguments = ["--windows-standalone-build", "--disable-auto-launch"] + COMMAND_LINE_ARGS_FOR_COMFYUI

	HAS_TORCH_CUDA : bool = check_python_torch_compiled_with_cuda(VENV_PYTHON_FILEPATH)
	if not HAS_TORCH_CUDA:
		update_python_torch_compiled_cuda(VENV_PYTHON_FILEPATH)
		HAS_TORCH_CUDA : bool = check_python_torch_compiled_with_cuda(VENV_PYTHON_FILEPATH)

	print("="*10)
	if HAS_TORCH_CUDA:
		print("If you want to run in LOW VRAM mode, enter 'yes'/'y', otherwise 'no'/'n'.")
		if input("").lower() in ("yes", "y"):
			print("Low VRAM")
			arguments.append("--lowvram")
	else:
		print("CPU Mode was selected.")
		arguments.append("--cpu")

	# start comfyui
	command1_args = [VENV_PYTHON_FILEPATH.as_posix(), (COMFYUI_DIRECTORY / "main.py").as_posix()] + arguments
	print("Running ComfyUI with the following commands:")
	print(command1_args)

	command2_args = [VENV_PYTHON_FILEPATH.as_posix(), (INSTALLER_DIRECTORY / "proxy.py").as_posix()]
	print("Running Proxy with the following commands:")
	print(command2_args)

	print("Starting threads...")
	thread1 = threading.Thread(target=lambda : run_command(command1_args))
	thread2 = threading.Thread(target=lambda : run_command(command2_args))
	thread3 = threading.Thread(target=check_for_proxy_and_comfyui_responses)
	thread1.start()
	thread2.start()
	thread3.start()
	thread1.join()
	thread2.join()
	thread3.join()

def main() -> None:
	print("Starting the install process for Windows!")
	print("- @spookexe was here")

	os.makedirs(TOOLS_DIRECTORY.as_posix(), exist_ok=True)

	print("[IMPORTANT]")
	print("The first generation may take a moment as it needs to load the AI models into memory.")
	print("[IMPORTANT]")
	print("Are you running the AI Image Generation with a AMD GPU? Use Task Manager to check your GPU 0.")
	print("Enter 'yes'/'y' if you are, otherwise 'no'/'n'.")
	if input("").lower() in ("y", "yes"):
		print("="*10)
		print("[NOTICE]")
		print("This version takes a minute to load due to dependencies.")
		print("You will also be required to do some manual installation later on.")
		print("="*10)
		comfyui_amd()
	else:
		comfyui_nvidia()

if __name__ == '__main__':
	main()
