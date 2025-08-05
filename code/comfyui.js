// Pure JavaScript implementation to connect to ComfyUI and generate images from a workflow.
// Assumes ComfyUI is running at the specified serverAddress (default: '127.0.0.1:8188').
// This code is designed to run in a browser environment. Note: Browser security policies
// may require ComfyUI to support CORS for HTTP requests and WebSocket connections.
// If CORS issues arise, ensure ComfyUI is configured accordingly or use a proxy.

// Helper function to generate a UUID (using crypto if available, fallback to simple random)
function generateUUID() {
	if (crypto && crypto.randomUUID) {
		return crypto.randomUUID().replace(/-/g, "");
	}
	return "xxxxxxxxxxxx4xxxyxxxxxxxxxxxxxxx".replace(/[xy]/g, function (c) {
		const r = (Math.random() * 16) | 0,
			v = c === "x" ? r : (r & 0x3) | 0x8;
		return v.toString(16);
	});
}

// Helper to convert blob to base64
async function blobToBase64(blob) {
	return new Promise((resolve, reject) => {
		const reader = new FileReader();
		reader.onloadend = () => resolve(reader.result.split(",")[1]); // Extract base64 part
		reader.onerror = reject;
		reader.readAsDataURL(blob);
	});
}

class ComfyUI_API {
	constructor(serverAddress = "127.0.0.1:8188") {
		this.serverAddress = serverAddress;
		this.clientId = generateUUID();
		this.activeIds = {};
		this.websocket = null;
	}

	async isAvailable() {
		try {
			const response = await fetch(`http://${this.serverAddress}`);
			if (!response.ok) {
				throw new Error("ComfyUI not available");
			}
		} catch (error) {
			throw new Error(`Cannot connect to ComfyUI: ${error.message}`);
		}
	}

	openWebSocket() {
		const address = `ws://${this.serverAddress}/ws?clientId=${this.clientId}`;
		this.websocket = new WebSocket(address);
		return new Promise((resolve, reject) => {
			this.websocket.onopen = resolve;
			this.websocket.onerror = reject;
		});
	}

	closeWebSocket() {
		if (this.websocket) {
			this.websocket.close();
			this.websocket = null;
		}
	}

	async queuePrompt(prompt) {
		const payload = { prompt, client_id: this.clientId };
		const response = await fetch(`http://${this.serverAddress}/prompt`, {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify(payload),
		});
		if (!response.ok) {
			throw new Error("Failed to queue prompt");
		}
		const data = await response.json();
		const promptId = data.prompt_id;
		if (!promptId) {
			throw new Error("Failed to get prompt_id from ComfyUI response");
		}
		this.activeIds[promptId] = false;
		return promptId;
	}

	isPromptIdFinished(promptId) {
		return this.activeIds[promptId];
	}

	async awaitPromptId(promptId) {
		let finished = this.isPromptIdFinished(promptId);
		while (finished === false) {
			await new Promise((resolve) => setTimeout(resolve, 1000));
			finished = this.isPromptIdFinished(promptId);
		}
		return finished;
	}

	async fetchPromptIdHistory(promptId) {
		const response = await fetch(
			`http://${this.serverAddress}/history/${promptId}`
		);
		if (!response.ok) {
			return null;
		}
		const history = await response.json();
		return history[promptId] || null;
	}

	async fetchImage(filename, subfolder, folderType) {
		const params = new URLSearchParams({
			filename,
			subfolder,
			type: folderType,
		});
		const response = await fetch(
			`http://${this.serverAddress}/view?${params}`
		);
		if (!response.ok) {
			throw new Error("Failed to fetch image");
		}
		return await response.blob();
	}

	async fetchPromptIdImages(promptId, includePreviews = false) {
		const images = [];
		const history = await this.fetchPromptIdHistory(promptId);
		if (!history) return images;

		for (const nodeId in history.outputs) {
			const output = history.outputs[nodeId];
			if (!output.images) continue;
			for (const image of output.images) {
				const outputData = {
					node_id: nodeId,
					file_name: image.filename,
					type: image.type,
				};
				if (
					image.type === "output" ||
					(includePreviews && image.type === "temp")
				) {
					const imageBlob = await this.fetchImage(
						image.filename,
						image.subfolder,
						image.type
					);
					outputData.image_data = imageBlob;
				}
				images.push(outputData);
			}
		}
		return images;
	}

	trackProgress(promptId, nodeIds) {
		return new Promise((resolve) => {
			const finishedNodes = [];
			this.websocket.onmessage = (event) => {
				if (typeof event.data !== "string") return;
				const message = JSON.parse(event.data);

				if (message.type === "progress") {
					const data = message.data;
					console.log(
						`In K-Sampler -> Step: ${data.value} of: ${data.max}`
					);
				}

				if (message.type === "execution_cached") {
					const data = message.data;
					for (const itm of data.nodes) {
						if (!finishedNodes.includes(itm)) {
							finishedNodes.push(itm);
							console.log(
								`Progress: ${finishedNodes.length - 1} / ${
									nodeIds.length
								} Tasks done`
							);
						}
					}
				}

				if (message.type === "executing") {
					const data = message.data;
					if (data.node && !finishedNodes.includes(data.node)) {
						finishedNodes.push(data.node);
						console.log(
							`Progress: ${finishedNodes.length - 1} / ${
								nodeIds.length
							} Tasks done`
						);
					}
					if (data.node === null && data.prompt_id === promptId) {
						this.activeIds[promptId] = true;
						resolve();
					}
				}

				if (message.type === "status") {
					if (message.data.status.exec_info.queue_remaining === 0) {
						console.log("Image is cached - breaking early.");
						resolve();
					}
				}
			};

			this.websocket.onerror = (error) => {
				console.error("WebSocket error:", error);
				resolve(); // Resolve to avoid hanging, but handle errors appropriately
			};
		});
	}

	cleanupPromptId(promptId) {
		delete this.activeIds[promptId];
	}

	async generateImagesUsingWorkflowPrompt(prompt, includePreviews = true) {
		const promptId = await this.queuePrompt(prompt);
		console.log("Track progress");
		await this.trackProgress(promptId, Object.keys(prompt));
		console.log("Fetching images from ComfyUI");
		const imageArray = await this.fetchPromptIdImages(
			promptId,
			includePreviews
		);
		console.log("Cleaning up prompt id");
		this.cleanupPromptId(promptId);
		return imageArray;
	}
}

async function generateWorkflowImage(
	workflow,
	serverAddress = "127.0.0.1:8188"
) {
	const comfyUI = new ComfyUI_API(serverAddress);
	await comfyUI.isAvailable();
	await comfyUI.openWebSocket();
	const imageArray = await comfyUI.generateImagesUsingWorkflowPrompt(
		workflow
	);
	comfyUI.closeWebSocket();

	// Convert blobs to base64 for easy use in webpage (e.g., img src="data:image/png;base64,...")
	const base64Images = [];
	for (const imgData of imageArray) {
		if (imgData.image_data) {
			const base64 = await blobToBase64(imgData.image_data);
			base64Images.push(base64);
		}
	}

	if (base64Images.length === 0) {
		throw new Error("ComfyUI failed to generate any image.");
	}

	console.log("Images were generated");
	return base64Images; // Returns array of base64 strings
}
