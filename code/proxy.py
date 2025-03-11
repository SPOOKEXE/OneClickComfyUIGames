
from PIL import Image
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
from pydantic import BaseModel
from typing import Any, Literal, Optional
from urllib.parse import urlencode
from uuid import uuid4

import base64
import aiohttp
import traceback
import requests
import websockets
import json
import asyncio
import aiohttp
import base64
import json
import traceback
import uvicorn
import websockets

PRINT_COMFYUI_INPUT_WORKFLOW : bool = False

COMFYUI_IMAGE_TYPE = Literal[
	"input", "output", "temp"
]

COMFYUI_SAMPLERS = Literal[
	"euler", "euler_ancestral", "heun",
	"dpm_2", "dpm_2_ancestral", "lms",
	"ddpm", "ddim", "uni_pc"
]

COMFYUI_SCHEDULERS= Literal[
	"normal", "karras", "exponential",
	"simple", "ddim_uniform", "beta"
]

def image_to_base64(image : Image.Image) -> str:
	"""Convert the PIL image back to a base64 image."""
	buffered = BytesIO()
	image.save(buffered, format="PNG")
	return base64.b64encode(buffered.getvalue()).decode("utf-8")

def base64_to_image(b64 : str) -> Image.Image:
	"""Convert the base64 image back to a PIL image."""
	buffer = BytesIO(base64.b64decode(b64))
	return Image.open(buffer).convert('RGB')

async def async_post(url : str, headers : Optional[dict] = None, cookies : Optional[dict] = None, json : Optional[dict] = None, data : Optional[str] = None) -> bytes:
	"""Asynchronously POST to the url and expect a json body response to return."""
	return requests.post(url, headers=headers, cookies=cookies, json=json, data=data).content

async def post_json_response(url : str, data : Optional[dict]) -> Optional[dict]:
	"""POST to the url and expect a json body response to return."""
	try:
		response : bytes = await async_post(url, json=data)
		return json.loads(response.decode('utf-8'))
	except Exception as e:
		traceback.print_exception(e)
		return None

async def async_get(url : str, headers : Optional[dict] = None, cookies : Optional[dict] = None, json : Optional[dict] = None, data : Optional[str] = None) -> bytes:
	"""Asynchronously GET to the url and expect a json body response to return."""
	return requests.get(url, headers=headers, cookies=cookies, json=json, data=data).content

async def get_json_response(url : str) -> Optional[dict]:
	"""GET to the url and expect a json body response to return."""
	try:
		response : bytes = await async_get(url)
		return json.loads(response.decode('utf-8'))
	except Exception as e:
		traceback.print_exception(e)
		return None

class ComfyUI_API:
	'''
	server_address : str = "127.0.0.1:8188"

	Cleaned up and asynchronous version of:
	- https://github.com/9elements/comfyui-api/blob/main/basic_api.py
	'''
	server_address : str
	client_id : str

	_active_ids : dict[str, bool]
	_websocket : Optional[Any]

	def __init__(self, server_address : str) -> None:
		self.server_address = server_address
		self.client_id = uuid4().hex
		self._active_ids = dict()

		# print(self.client_id)

	async def is_available(self) -> None:
		try:
			_ = await async_get(f"http://{self.server_address}")
		except:
			raise Exception("Cannot connect to ComfyUI!")

	async def open_websocket(self) -> None:
		address : str = f"ws://{self.server_address}/ws?clientId={self.client_id}"
		self._websocket = websockets.connect(address)

	async def close_websocket(self) -> None:
		self._websocket = None

	async def queue_prompt(self, prompt : dict) -> str:
		'''Queue the given prompt and return a prompt_id'''
		payload = {"prompt": prompt, "client_id": self.client_id}
		# print(payload)
		response : Optional[dict] = await post_json_response(f"http://{self.server_address}/prompt", data=payload)
		# print(response)
		assert response, "ComfyUI is not currently running."
		prompt_id : Optional[str] = response.get('prompt_id')
		# print(prompt_id)
		assert prompt_id, "Failed to get prompt_id from ComfyUI response."
		# print(prompt_id)
		self._active_ids[prompt_id] = False
		return prompt_id

	def is_prompt_id_finished(self, prompt_id : str) -> Optional[bool]:
		'''Check if the given prompt_id is finished.'''
		return self._active_ids.get(prompt_id)

	async def await_prompt_id(self, prompt_id : str) -> Optional[bool]:
		'''Await for the prompt id to finish - also returns if it finished or not.'''
		finished : Optional[bool] = self.is_prompt_id_finished(prompt_id)
		while finished is False:
			await asyncio.sleep(1.0)
			finished = self.is_prompt_id_finished(prompt_id)
		return finished

	async def fetch_prompt_id_history(self, prompt_id : str) -> Optional[dict]:
		'''Fetch the generation history for the given prompt_id.'''
		history = await get_json_response(f"http://{self.server_address}/history/{prompt_id}")
		if history is None:
			return None
		return history.get(prompt_id)

	async def fetch_image(self, filename : str, subfolder : str, folder_type : str) -> bytes:
		payload = {"filename": filename, "subfolder": subfolder, "type": folder_type}
		query : str = urlencode(payload)
		response : bytes = await async_get(f"http://{self.server_address}/view?{query}")
		return response

	async def fetch_preview_images(self, prompt_id : str) -> list[dict]:
		'''Fetch the generated images for the given prompt_id if any.'''
		images : list[dict] = list()
		history : Optional[dict] = await self.fetch_prompt_id_history(prompt_id)
		if history is None: return []
		for node_id in history['outputs']:
			github_spookexe_was_here = history['outputs'][node_id]
			if 'images' not in github_spookexe_was_here:
				continue
			for image in github_spookexe_was_here['images']:
				output_data = {"node_id" : node_id, "file_name" : image["filename"], "type" : image["type"]}
				if image['type'] == 'temp':
					preview_data : bytes = await self.fetch_image(image['filename'], image['subfolder'], image['type'])
					output_data['image_data'] = preview_data
				images.append(output_data)
		return images

	async def fetch_prompt_id_images(self, prompt_id : str, include_previews : bool = False) -> list[dict]:
		'''Fetch the generated images for the given prompt_id if any.'''
		images : list[dict] = list()
		history : Optional[dict] = await self.fetch_prompt_id_history(prompt_id)
		if history is None: return []
		for node_id in history['outputs']:
			github_spookexe_was_here = history['outputs'][node_id]
			if 'images' not in github_spookexe_was_here:
				continue
			for image in github_spookexe_was_here['images']:
				output_data = {"node_id" : node_id, "file_name" : image["filename"], "type" : image["type"]}
				if include_previews is True and image['type'] == 'temp':
					preview_data : bytes = await self.fetch_image(image['filename'], image['subfolder'], image['type'])
					output_data['image_data'] = preview_data
				if image['type'] == 'output':
					image_data : bytes = await self.fetch_image(image['filename'], image['subfolder'], image['type'])
					output_data['image_data'] = image_data
				images.append(output_data)
		return images

	async def track_progress(self, prompt_id : str, node_ids : list[int]) -> None:
		'''Echo the progress of the prompt_id.'''
		finished_nodes : list[str] = []
		assert self._websocket, "No websocket is currently connected to ComfyUI."
		async with self._websocket as socket:
			while True:
				# receive content
				content = await socket.recv()
				if isinstance(content, str) is False:
					continue
				# print(content)
				message = json.loads(content)
				# progression of current
				if message['type'] == 'progress':
					data = message['data']
					current_step = data['value']
					print('In K-Sampler -> Step: ', current_step, ' of: ', data['max'])
				# another step of execution done
				if message['type'] == 'execution_cached':
					spookexe_github_was_here = message['data']
					for itm in spookexe_github_was_here['nodes']:
						if itm not in finished_nodes:
							finished_nodes.append(itm)
							print('Progess: ', len(finished_nodes)-1, '/', len(node_ids), ' Tasks done')
				# executing a new node if any
				if message['type'] == 'executing':
					data = message['data']
					if data['node'] not in finished_nodes:
						finished_nodes.append(data['node'])
						print('Progess: ', len(finished_nodes)-1, '/', len(node_ids), ' Tasks done')
					if data['node'] is None and data['prompt_id'] == prompt_id:
						# execution is done
						self._active_ids[prompt_id] = True
						break
				# check if status is queue empty (for if its already cached)
				if message['type'] == "status":
					if message["data"]["status"]["exec_info"]["queue_remaining"] == 0:
						print("Image is cached - breaking early.")
						break # already finished

	async def cleanup_prompt_id(self, prompt_id : str) -> None:
		self._active_ids.pop(prompt_id, None)

	async def generate_images_using_workflow_prompt(self, prompt : dict, include_previews : bool = True) -> list[dict]:
		'''Complete the full sequence of giving a prompt and receiving the images.'''
		prompt_id : str = await self.queue_prompt(prompt)
		print('Track progress')
		await self.track_progress(prompt_id, list(prompt.keys()))
		print('Fetching images from ComfyUI')
		images_spookexe : list[dict] = await self.fetch_prompt_id_images(prompt_id, include_previews=include_previews)
		print('Cleaning up prompt id')
		await self.cleanup_prompt_id(prompt_id)
		return images_spookexe

	async def upload_image(self, image : Image.Image, save_name : str, image_type : COMFYUI_IMAGE_TYPE = "input", overwrite : bool = True) -> bool:
		"""Upload an image to ComfyUI to be used for workflows."""
		# use 'save_name' in LoadImage objects (with extension)
		# prepare image data
		byte_io = BytesIO()
		image.save(byte_io, format='PNG')
		byte_data = byte_io.getvalue()
		# prepare form data
		data = aiohttp.FormData()
		data.add_field('image', byte_data, filename=save_name, content_type="image/png")
		data.add_field('type', image_type)
		data.add_field('overwrite', str(overwrite).lower())
		# send request
		try:
			url : str = f'http://{self.server_address}/upload/image'
			client : aiohttp.ClientSession
			async with aiohttp.ClientSession() as client:
				response : aiohttp.ClientResponse = await client.post(url, data=data)
				if response.status != 200:
					print(f"Failed to upload image due to: {response.reason} (typicallycfile data)")
			return True
		except Exception as e:
			traceback.print_exception(e)
			return False

COMFYUI_NODE_URL : str = '127.0.0.1:8188' # change this to redirect

class GenerateImagesResponse(BaseModel):
	"""What is returned from the proxy /generate_workflow api."""
	images : list[str]

async def generate_workflow_image(workflow : dict) -> list[Image.Image]:
	"""Process a workflow and return the first image."""
	COMFYUI_NODE = ComfyUI_API(COMFYUI_NODE_URL)
	await COMFYUI_NODE.is_available()
	await COMFYUI_NODE.open_websocket()
	image_array : list[dict] = await COMFYUI_NODE.generate_images_using_workflow_prompt(workflow)
	await COMFYUI_NODE.close_websocket()
	return [Image.open(BytesIO(image_data["image_data"])) for image_data in image_array]

app = FastAPI(title='Local Image Generation', description='This api allows local image generation with ComfyUI. Coded by @SPOOKEXE on GitHub', version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.get('/echo', description='Echo back to let the client know the api is running.')
async def echo() -> bool:
	return True

@app.post('/generate_workflow', description='Generate a image given the generation workflow.')
async def generate_worflow(workflow : dict) -> Optional[GenerateImagesResponse]:
	if PRINT_COMFYUI_INPUT_WORKFLOW:
		print(workflow)
	try:
		image_pil : list[Image.Image] = await generate_workflow_image(workflow)
		image_bs4 : list[str] = [image_to_base64(pil_image) for pil_image in image_pil]
		assert len(image_bs4) != 0, "ComfyUI failed to generate any image."
		print('Images were generated, sending results to Abyss Diver')
		return GenerateImagesResponse(images=image_bs4)
	except Exception as e:
		print(e)
	return None

async def uvicorn_run(app : FastAPI, host : str = "127.0.0.1", port : int = 12500) -> None:
	config = uvicorn.Config(app, host=host, port=port, access_log=False, server_header=True, date_header=False, proxy_headers=False)
	await uvicorn.Server(config).serve()

if __name__ == '__main__':
	asyncio.run(uvicorn_run(app, host='127.0.0.1', port=12500))
