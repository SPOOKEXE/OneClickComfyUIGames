# /// script
# requires-python = ">=3.12"
# dependencies = ["tqdm", "requests"]
# ///

'''
Process (NVIDIA and AMD):
1. git clone ComfyUI
2. uv init && uv venv
3. uv add ...
4. download custom_nodes
5. uv add ...
6. download Checkpoints to models/checkpoints & models/loras
7. run comfyui
'''

'''
Links:
- ComfyUI: https://github.com/Comfy-Org/ComfyUI?tab=readme-ov-file#manual-install-windows-linux
- Checkpoint (PONY): https://civitai.com/models/376031/hassaku-xl-hentai
- LORA: https://civitai.com/models/481529/dall-e-3-anime-style-pony
'''

from pathlib import Path
import shutil
from typing import Any, List, Tuple, Dict, Optional

import traceback
import logging
import json
import sys
import platform
import os
import subprocess
import threading

INSTALLER_DIRECTORY: Path = Path(os.path.dirname(os.path.abspath(__file__)))
TOOLS_DIRECTORY: Path = INSTALLER_DIRECTORY / "tools"

COMFYUI_MAIN_REPOSITORY_URL: str = "https://github.com/comfyanonymous/ComfyUI"

COMMAND_LINE_ARGS_FOR_COMFYUI: List[str] = [
	# custom command line arguments for comfyui - separate each string by comma
	# e.g. ["--listen", "--api", "--highvram", "--somevalue", "5"]
	"--enable-cors-header", "*", "--lowvram"
]

COMFYUI_CUSTOM_NODES: List[str] = [
	"https://github.com/ltdrdata/ComfyUI-Manager",
	"https://github.com/john-mnz/ComfyUI-Inspyrenet-Rembg"
]

AI_CHECKPOINT_DOWNLOADS: Dict[str, Optional[str]] = {
	# PonyXL
	"hassakuXLPony_v13BetterEyesVersion.safetensors" : "https://huggingface.co/FloricSpacer/AbyssDiverModels/resolve/main/hassakuXLPony_v13BetterEyesVersion.safetensors?download=true"
}

AI_LORA_DOWNLOADS: Dict[str, Optional[str]] = {
	# PonyXL
	"DallE3-magik.safetensors" : "https://huggingface.co/FloricSpacer/AbyssDiverModels/resolve/main/DallE3-magik.safetensors?download=true"
}

WINDOWS_TORCH_CUDA_INDEX_URL: str = "https://download.pytorch.org/whl/cu130"
WINDOWS_TORCH_ROCM_INDEX_URL: str = "https://download.pytorch.org/whl/rocm6.4"
WINDOWS_TORCH_INTEL_INDEX_URL: str = "https://download.pytorch.org/whl/xpu"
LINUX_TORCH_CUDA_INDEX_URL: str = "https://download.pytorch.org/whl/cu130"
LINUX_TORCH_ROCM_INDEX_URL: str = "https://download.pytorch.org/whl/rocm6.4"
LINUX_TORCH_INTEL_INDEX_URL: str = "https://download.pytorch.org/whl/xpu"
MAC_PRERELEASE_INDEX_URL: str = "https://download.pytorch.org/whl/nightly/cpu"

class CommandsManager:
	"""Commands manager and running - logs everything that runs"""

	def run_command(args: List[str], **kwargs) -> int:
		print("run_command", args, dict(kwargs))
		return subprocess.call(args, **kwargs, shell=False)

	def run_process(args: List[str], **kwargs) -> Tuple[int, str]:
		print("run_process", args, dict(kwargs))
		try:
			kwarg_dict = dict(kwargs)
			# force certain arguments
			kwarg_dict['shell'] = False
			kwarg_dict['stdout'] = subprocess.PIPE
			kwarg_dict['stderr'] = subprocess.PIPE
			kwarg_dict['text'] = True
			kwarg_dict['bufsize'] = 1
			process = subprocess.Popen(args, **kwarg_dict)
			stdout_var : str = ""
			def stream_reader(pipe, log_level):
				nonlocal stdout_var
				"""Reads from a pipe and logs each line at the given log level."""
				with pipe:
					for line in iter(pipe.readline, ''):
						print(log_level, line.strip())
						stdout_var += line.strip()
			# Use threads to prevent blocking
			stdout_thread = threading.Thread(
				target=stream_reader, args=(process.stdout, logging.INFO)
			)
			stderr_thread = threading.Thread(
				target=stream_reader, args=(process.stderr, logging.ERROR)
			)
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

class FFlagManager:
	"""FFlag manager (to mark what was done)"""

	@staticmethod
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

	@staticmethod
	def get_fflag(key : str) -> Any:
		fflags = FFlagManager.get_fflags()
		return fflags.get(key, False)

	@staticmethod
	def set_fflag(key : str, value : Any):
		fflags = FFlagManager.get_fflags()
		fflags[key] = value
		with open("fflag", "w", encoding='utf-8', errors='ignore') as file:
			file.write(json.dumps(fflags, indent=4))

class WindowsMisc:
	"""Methods specific to Windows"""

	@staticmethod
	def assert_path_length_limit() -> None:
		"""Check how long the path is for the local-gen folder."""
		current_path : str = Path(os.path.abspath(os.getcwd())).as_posix()
		path_length : int = len(current_path) + 70 # 70 being approx submodules of ComfyUI
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

def download_file(url: str, filepath: Path, chunk_size: int = 64) -> None:
	"""Download file from the url to the filepath, chunk_size is how much data is downloaded at once in the stream."""
	from tqdm import tqdm
	import requests
	response = requests.get(url, stream=True) # type: ignore
	response.raise_for_status()  # Raise an error for bad status codes
	total_size = int(response.headers.get('content-length', 0))
	progress_bar = tqdm(total=total_size, unit='B', unit_scale=True, desc=filepath.name) # type: ignore
	with open(filepath, 'wb') as file:
		for chunk in response.iter_content(chunk_size=chunk_size * 1024):
			file.write(chunk)
			progress_bar.update(len(chunk))
	progress_bar.close()
	print(f"File downloaded to {filepath.as_posix()}")

def download_file_wrapper(folder: Path, filename: str, url: Optional[str]):
	if url is None:
		print(f'File {filename} has no download url.')
		return
	filepath = folder / "checkpoints" / filename
	if filepath.exists():
		print(f"File {filename} is already installed.")
		return
	try:
		download_file(url, filepath)
	except Exception as e:
		print(f'Failed to download file {filename}.')
		traceback.print_exception(e)
		raise Exception(f"Failed to download the {filename} file.")

def download_checkpoints_to_subfolder(models_folder: Path) -> None:
	"""Download the checkpoints to the sub-folder checkpoints"""
	for filename, url in AI_CHECKPOINT_DOWNLOADS.items():
		download_file_wrapper(models_folder / "checkpoints", filename, url)

def download_loras_to_subfolder(models_folder: Path) -> None:
	"""Download the checkpoints to the sub-folder loras"""
	for filename, url in AI_LORA_DOWNLOADS.items():
		download_file_wrapper(models_folder / "loras", filename, url)

def get_compute_device() -> int:
	# 0:cpu, 1:cuda, 2:amd, 3:intel gpu
	option: int | bool = FFlagManager.get_fflag("gpu")
	if option is not False: return int(option)
	print("="*10)
	print("You will be selecting your compute device.")
	print("Windows: You can check this information in the TASK MANAGER in the performance tab.")
	print("Linux/Mac: `lshw -C display | grep product` in the terminal.")
	print("Available options are: CPU, NVIDIA, AMD and INTEL. They will pop up one by one.")
	print("")
	if input("Are you going to generate on the CPU? (y/n) ").lower() == "y":
		FFlagManager.set_fflag("gpu", 0)
		return 0
	if input("Are you going to generate on a NVIDIA GPU? (y/n) ").lower() == "y":
		FFlagManager.set_fflag("gpu", 1)
		return 1
	if input("Are you going to generate on a AMD GPU? (y/n) ").lower() == "y":
		FFlagManager.set_fflag("gpu", 2)
		return 2
	if input("Are you going to generate on a Intel GPU? (y/n) ").lower() == "y":
		FFlagManager.set_fflag("gpu", 3)
		return 3
	print("Defaulting to the CPU.")
	FFlagManager.set_fflag("gpu", 0)
	return 0

def clone_custom_nodes_to_folder(folder: Path) -> None:
	failed_clone: List[str] = []
	os.chdir(folder.as_posix())
	for repo_url in COMFYUI_CUSTOM_NODES:
		folder_name: str = repo_url.rstrip("/").split("/")[-1]
		print(f'Cloning repository {repo_url} to {folder_name}')
		try:
			_ = CommandsManager.run_command(["git", "clone", repo_url])
		except Exception as e:
			traceback.print_exception(e)
			print(f"Failed to clone repository {repo_url}")
			failed_clone.append(repo_url)
		repo_folder_path: Path = folder / folder_name
		if failed_clone and repo_folder_path.exists():
			shutil.rmtree(repo_folder_path)
	if failed_clone:
		print("Failed to clone repositories:")
		print("\n".join(failed_clone))
		print(f"Either run the script again when internet is stable, or manually do it with `git clone [url]` in {folder.as_posix()}.")
	os.chdir(INSTALLER_DIRECTORY)

def install_comfyui_requirements_shared(folder: Path) -> None:
	"""For post-torch install"""
	comfyui_directory: Path = folder / "ComfyUI"
	custom_nodes_folder: Path = comfyui_directory / "custom_nodes"

	os.chdir(comfyui_directory.as_posix())

	# requirements from main
	requirements_file = comfyui_directory / "requirements.txt"
	_ = CommandsManager.run_command(["uv", "add", "-r", requirements_file.as_posix()])

	# requirements from custom nodes
	for repo_url in COMFYUI_CUSTOM_NODES:
		folder_name: str = repo_url.rstrip("/").split("/")[-1]

		requirements_file = custom_nodes_folder / folder_name / "requirements.txt"
		if not requirements_file.exists():
			continue
		print(f"Found custom nodes requirements file - installing requirements.")
		print(requirements_file.as_posix())
		_ = CommandsManager.run_command(["uv", "add", "-r", requirements_file.as_posix()])

	os.chdir(INSTALLER_DIRECTORY)

def install_comfyui_shared(folder: Path) -> None:

	comfyui_directory = folder / "ComfyUI"
	print(f'ComfyUI install directory: {comfyui_directory.as_posix()}')

	# clone comfyui
	if not comfyui_directory.exists():
		print(f"Attempting to clone ComfyUI into {comfyui_directory.as_posix()}")
		os.chdir(folder)
		try:
			status = CommandsManager.run_command(["git", "clone", COMFYUI_MAIN_REPOSITORY_URL])
			print(status)
		except Exception as e:
			traceback.print_exception(e)
			status = 1
		# if failed to clone fully, delete the partial-cloned folder
		if status != 0 and comfyui_directory.exists():
			print("ComfyUI only partially cloned, deleting folder.")
			shutil.rmtree(comfyui_directory.as_posix())
		assert status == 0, "git clone has failed!"
		os.chdir(INSTALLER_DIRECTORY)

	print("Checking project toml for dependencies...")
	proj_toml: Path = comfyui_directory / "pyproject.toml"
	if not proj_toml.exists():
		os.chdir(comfyui_directory)
		CommandsManager.run_command(["uv", "init"])
		os.chdir(INSTALLER_DIRECTORY)

	# setup venv
	venv_folder = comfyui_directory / ".venv"
	if not venv_folder.exists():
		print(f"Setting up uv project in {comfyui_directory.as_posix()}")
		os.chdir(comfyui_directory)
		_ = CommandsManager.run_command(["uv", "venv"])
		# install basic requirements
		_ = CommandsManager.run_command(["uv", "add", "pip", "--default-index", "https://pypi.org/simple"])
		os.chdir(INSTALLER_DIRECTORY)

	# clone custom node
	custom_nodes_folder = comfyui_directory / "custom_nodes"
	clone_custom_nodes_to_folder(custom_nodes_folder)

def run_windows_installer() -> None:

	install_comfyui_shared(TOOLS_DIRECTORY)

	comfyui_directory = TOOLS_DIRECTORY / "ComfyUI"
	os.chdir(comfyui_directory)

	args = ["uv", "run", "main.py"]

	compute_device: int = get_compute_device()
	if compute_device == 0:
		print("CPU")
		# pip install torch torchvision torchaudio
		_ = CommandsManager.run_command(["uv", "add", "torch", "torchvision", "torchaudio"])
	elif compute_device == 1:
		print("CUDA")
		# pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu130
		_ = CommandsManager.run_command(["uv", "add", "torch", "torchvision", "torchaudio", "--index", WINDOWS_TORCH_CUDA_INDEX_URL])
	elif compute_device == 2:
		print("AMD")
		# pip install torch torchvision torchaudio --index https://download.pytorch.org/whl/rocm6.4
		_ = CommandsManager.run_command(["uv", "add", "torch", "torchvision", "torchaudio", "--index", WINDOWS_TORCH_ROCM_INDEX_URL])
	elif compute_device == 3:
		print("INTEL")
		# pip install torch torchvision torchaudio --index https://download.pytorch.org/whl/xpu
		_ = CommandsManager.run_command(["uv", "add", "torch", "torchvision", "torchaudio", "--index", WINDOWS_TORCH_INTEL_INDEX_URL])

	os.chdir(INSTALLER_DIRECTORY)

	install_comfyui_requirements_shared(TOOLS_DIRECTORY)

	os.chdir(comfyui_directory)

	print("-"*10)
	print("Running ComfyUI...")
	CommandsManager.run_process(args + COMMAND_LINE_ARGS_FOR_COMFYUI)

def run_mac_installer() -> None:
	install_comfyui_shared(TOOLS_DIRECTORY)

	comfyui_directory = TOOLS_DIRECTORY / "ComfyUI"
	os.chdir(comfyui_directory)

	args = ["uv", "run", "main.py"]

	# pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
	_ = CommandsManager.run_command(["uv", "add", "--prerelease", "explicit", "torch", "torchvision", "torchaudio", "--index", MAC_PRERELEASE_INDEX_URL])

	os.chdir(INSTALLER_DIRECTORY)

	install_comfyui_requirements_shared(TOOLS_DIRECTORY)

	os.chdir(comfyui_directory)

	print("-"*10)
	print("Running ComfyUI...")
	CommandsManager.run_process(args + COMMAND_LINE_ARGS_FOR_COMFYUI)

def run_linux_installer() -> None:
	install_comfyui_shared(TOOLS_DIRECTORY)

	comfyui_directory = TOOLS_DIRECTORY / "ComfyUI"
	os.chdir(comfyui_directory)

	args = ["uv", "run", "main.py"]

	compute_device: int = get_compute_device()
	if compute_device == 0:
		print("CPU")
		# pip install torch torchvision torchaudio
		_ = CommandsManager.run_command(["uv", "add", "torch", "torchvision", "torchaudio"])
	elif compute_device == 1:
		print("CUDA")
		# pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu130
		_ = CommandsManager.run_command(["uv", "add", "torch", "torchvision", "torchaudio", "--index", WINDOWS_TORCH_CUDA_INDEX_URL])
	elif compute_device == 2:
		print("AMD")
		# pip install torch torchvision torchaudio --index https://download.pytorch.org/whl/rocm6.4
		_ = CommandsManager.run_command(["uv", "add", "torch", "torchvision", "torchaudio", "--index", WINDOWS_TORCH_ROCM_INDEX_URL])
	elif compute_device == 3:
		print("INTEL")
		# pip install torch torchvision torchaudio --index https://download.pytorch.org/whl/xpu
		_ = CommandsManager.run_command(["uv", "add", "torch", "torchvision", "torchaudio", "--index", WINDOWS_TORCH_INTEL_INDEX_URL])

	os.chdir(INSTALLER_DIRECTORY)

	install_comfyui_requirements_shared(TOOLS_DIRECTORY)

	os.chdir(comfyui_directory)

	print("-"*10)
	print("Running ComfyUI...")
	CommandsManager.run_process(args + COMMAND_LINE_ARGS_FOR_COMFYUI)


def main() -> None:
	if platform.system() == "Windows":
		print("Checking path length limit for Windows")
		WindowsMisc.assert_path_length_limit()

	# try run with the libraries, otherwise restart script and check again
	print("Checking requests and tqdm availability...")
	try:
		import requests
		import tqdm
	except ImportError:
		if sys.argv[1] == "1":
			print("requests and tqdm were not installed even after script restart!")
			print(f"Try manually install it by opening a terminal in the {INSTALLER_DIRECTORY.as_posix()} directory and doing `uv init && uv add requests tqdm`.")
			exit(1)
		print("requests and tqdm were not successfully hot loaded in, attempting script restart")
		status = CommandsManager.run_process("uv run --script installer.py 1")
		exit(status)

	os.makedirs(TOOLS_DIRECTORY, exist_ok=True)

	# Now run installer specific
	if platform.system() == "Windows":
		print("Running Windows.")
		run_windows_installer()
	elif platform.system() == "Linux":
		print('Running Linux.')
		run_linux_installer()
	elif platform.system() == "Darwin":
		print('Running Mac.')
		run_mac_installer()

if __name__ == '__main__':
	main()
