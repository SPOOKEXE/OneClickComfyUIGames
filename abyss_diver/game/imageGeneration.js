
/**
 * @type {object}
 * @property {Object.<string, any>} variables
 */
State

/**
 * @callback variablesGetter
 * @returns {Object.<string, any>}
 */
/**
 * @type {variablesGetter}
 */
variables

// This JSDoc declaration defines the extensions we add to setup in this file
/**
 * @typedef SugarCubeSetupObject
 * @extends SugarCubeSetupObject
 * @property {string} Path
 * @property {string} ImagePath
 * @property {string} SoundPath
 * @property {int} never
 * @property {string[]} flaskLabels
 * @property {function} curse
 * @property {function} curses
 * @property {function} item
 * @property {function} items
 * @property {function} relic
 * @property {function} relics
 * @property {function} companion
 * @property {function} companions
 * @property {function} sellValue
 * @property {function} modAffection
 * @property {function} activeCurseCount
 * @property {number} carriedWeight
 * @property {boolean} haveCuttingTool
 * @property {boolean} haveSword
 * @property {boolean} haveScubaGear
 * @property {boolean} haveSmartphoneRegular
 * @property {boolean} haveSmartphoneAI
 * @property {boolean} haveSmartphone
 * @property {boolean} haveUnlimitedLightSource
 * @property {boolean} havePotentialLightSource
 * @property {boolean} haveTravelLightSource
 * @property {boolean} haveNotepad
 * @property {boolean} haveRope
 * @property {boolean} haveHealing
 * @property {boolean} haveColdProtection
 * @property {boolean} haveHeatProtection
 * @property {function} sellRelic
 * @property {function} unsellRelic
 * @property {function} loseRelic
 * @property {boolean} passingTime
 * @property {function} startPassingTime
 * @property {function} stopPassingTime
 * @property {function} isPregnant
 * @property {function} setConsideredPregnant
 * @property {function} setNotPregnant
 * @property {function} dueDate
 * @property {function} daysConsideredPregnant
 * @property {function} daysUntilDue
 * @property {function} willingCurses
 * @property {function} returnRelic
 * @property {function} getCurseSets
 * @property {function} getUserCurseSets
 * @property {function} setUserCurseSets
 * @property {function} addUserCurseSet
 */

/*
	===============================================
	WORKFLOWS
	===============================================
*/

const SIMPLE_T2I_PORTRAIT_WORKFLOW = {
	"5": {
		"inputs": {"ckpt_name": "sd_xl_base_1.0.safetensors"},
		"class_type": "CheckpointLoaderSimple",
		"_meta": {"title": "Load Checkpoint"}
	},
	"9": {
		"inputs": {"text": "","clip": ["5",1]},
		"class_type": "CLIPTextEncode",
		"_meta": {"title": "Positive Prompt"}
	},
	"10": {
		"inputs": {"text": "","clip": ["5",1]},
		"class_type": "CLIPTextEncode",
		"_meta": {"title": "Negative Prompt"}
	},
	"11": {
		"inputs": {
			"seed": 0,
			"steps": 30,
			"cfg": 9,
			"sampler_name": "euler",
			"scheduler": "normal",
			"denoise": 1,
			"model": ["5",0],
			"positive": ["9",0],
			"negative": ["10",0],
			"latent_image": ["12",0]
		},
		"class_type": "KSampler",
		"_meta": {"title": "KSampler"}
	},
	"12": {
		"inputs": {"width": 1024,"height": 1024,"batch_size": 1},
		"class_type": "EmptyLatentImage",
		"_meta": {"title": "Empty Latent Image"}
	},
	"15": {
		"inputs": {"samples": ["11",0],"vae": ["5",2]},
		"class_type": "VAEDecode",
		"_meta": {"title": "VAE Decode"}
	},
	"16": {
		"inputs": {"filename_prefix": "ComfyUI","images": ["15",0]},
		"class_type": "SaveImage",
		"_meta": {"title": "Save Image"}
	}
}

const SIMPLE_T2I_PORTRAIT_WORKFLOW_REMOVE_BACKGROUND = {
	"5": {
		"inputs": {
			"ckpt_name": "hassakuXLPony_v13BetterEyesVersion.safetensors"
		},
		"class_type": "CheckpointLoaderSimple",
		"_meta": {
			"title": "Load Checkpoint"
		}
	},
	"9": {
		"inputs": {
			"text": "",
			"clip": [
				"5",
				1
			]
		},
		"class_type": "CLIPTextEncode",
		"_meta": {
			"title": "Positive Prompt"
		}
	},
	"10": {
		"inputs": {
			"text": "",
			"clip": [
				"5",
				1
			]
		},
		"class_type": "CLIPTextEncode",
		"_meta": {
			"title": "Negative Prompt"
		}
	},
	"11": {
		"inputs": {
			"seed": -1,
			"steps": 35,
			"cfg": 9,
			"sampler_name": "euler",
			"scheduler": "normal",
			"denoise": 1,
			"model": [
				"5",
				0
			],
			"positive": [
				"9",
				0
			],
			"negative": [
				"10",
				0
			],
			"latent_image": [
				"12",
				0
			]
		},
		"class_type": "KSampler",
		"_meta": {
			"title": "KSampler"
		}
	},
	"12": {
		"inputs": {
			"width": 1024,
			"height": 1024,
			"batch_size": 1
		},
		"class_type": "EmptyLatentImage",
		"_meta": {
			"title": "Empty Latent Image"
		}
	},
	"15": {
		"inputs": {
			"samples": [
				"11",
				0
			],
			"vae": [
				"5",
				2
			]
		},
		"class_type": "VAEDecode",
		"_meta": {
			"title": "VAE Decode"
		}
	},
	"17": {
		"inputs": {
			"threshold": 0.85,
			"torchscript_jit": "default",
			"image": [
				"15",
				0
			]
		},
		"class_type": "InspyrenetRembgAdvanced",
		"_meta": {
			"title": "Inspyrenet Rembg Advanced"
		}
	},
	"18": {
		"inputs": {
			"filename_prefix": "TRANSPARENT_",
			"images": [
				"17",
				0
			]
		},
		"class_type": "SaveImage",
		"_meta": {
			"title": "Save Image"
		}
	}
}

const SCENE_GENERATION_WITH_MIDAS_WORKFLOW = {
	"1": {
		"inputs": {"image": "00007-4032934194.png","upload": "image"},
		"class_type": "LoadImage",
		"_meta": {"title": "Load Image"}
	},
	"2": {
		"inputs": {"a": 6.2832,"bg_threshold": 0.1,"resolution": 512,"image": ["1",0]},
		"class_type": "MiDaS-DepthMapPreprocessor",
		"_meta": {"title": "MiDaS Depth Map"}
	},
	"3": {
		"inputs": {"images": ["2",0]},
		"class_type": "PreviewImage",
		"_meta": {"title": "Preview Image"}
	},
	"5": {
		"inputs": {"ckpt_name": "ThinkDiffusionXL.safetensors"},
		"class_type": "CheckpointLoaderSimple",
		"_meta": {"title": "Load Checkpoint"}
	},
	"7": {
		"inputs": {"control_net_name": "control-lora-depth-rank256.safetensors"},
		"class_type": "ControlNetLoader",
		"_meta": {"title": "Load ControlNet Model"}
	},
	"8": {
		"inputs": {
			"strength": 1,
			"start_percent": 0,
			"end_percent": 1,
			"positive": ["9",0],
			"negative": ["10",0],
			"control_net": ["7",0],
			"image": ["2",0]
		},
		"class_type": "ControlNetApplyAdvanced",
		"_meta": {"title": "Apply ControlNet (Advanced)"}
	},
	"9": {
		"inputs": {"text": "a futuristic cyborg on an alien spaceship","clip": ["5",1]},
		"class_type": "CLIPTextEncode",
		"_meta": {"title": "Positive Prompt"}
	},
	"10": {
		"inputs": {"text": "","clip": ["5",1]},
		"class_type": "CLIPTextEncode",
		"_meta": {"title": "Negative Prompt"}
	},
	"11": {
		"inputs": {
			"seed": 961176784184834,
			"steps": 35,
			"cfg": 7,
			"sampler_name": "dpmpp_3m_sde_gpu",
			"scheduler": "exponential",
			"denoise": 1,
			"model": ["5",0],
			"positive": ["8",0],
			"negative": ["8",1],
			"latent_image": ["12",0]
		},
		"class_type": "KSampler",
		"_meta": {"title": "KSampler"}
	},
	"12": {
		"inputs": {"width": 1024,"height": 1024,"batch_size": 1},
		"class_type": "EmptyLatentImage",
		"_meta": {"title": "Empty Latent Image"}
	},
	"15": {
		"inputs": {"samples": ["11",0], "vae": ["5",2]},
		"class_type": "VAEDecode",
		"_meta": {"title": "VAE Decode"}
	},
	"16": {
		"inputs": {"filename_prefix": "ComfyUI", "images": ["15",0]},
		"class_type": "SaveImage",
		"_meta": {"title": "Save Image"}
	}
}

const SCENE_GENERATION_WITHOUT_MIDAS_WORKFLOW = {
	"1": {
		"inputs": {"image": "00007-4032934194.png","upload": "image"},
		"class_type": "LoadImage",
		"_meta": {"title": "Load Image"}
	},
	"3": {
		"inputs": {"images": ["1",0]},
		"class_type": "PreviewImage",
		"_meta": {"title": "Preview Image"}
	},
	"5": {
		"inputs": {"ckpt_name": "ThinkDiffusionXL.safetensors"},
		"class_type": "CheckpointLoaderSimple",
		"_meta": {"title": "Load Checkpoint"}
	},
	"7": {
		"inputs": {"control_net_name": "control-lora-depth-rank256.safetensors"},
		"class_type": "ControlNetLoader",
		"_meta": {"title": "Load ControlNet Model"}
	},
	"8": {
		"inputs": {
			"strength": 1,
			"start_percent": 0,
			"end_percent": 1,
			"positive": ["9",0],
			"negative": ["10",0],
			"control_net": ["7",0],
			"image": ["1",0]
		},
		"class_type": "ControlNetApplyAdvanced",
		"_meta": {"title": "Apply ControlNet (Advanced)"}
	},
	"9": {
		"inputs": {"text": "a futuristic cyborg on an alien spaceship","clip": ["5",1]},
		"class_type": "CLIPTextEncode",
		"_meta": {"title": "Positive Prompt"}
	},
	"10": {
		"inputs": {"text": "","clip": ["5",1]},
		"class_type": "CLIPTextEncode",
		"_meta": {"title": "Negative Prompt"}
	},
	"11": {
		"inputs": {
			"seed": 961176784184834,
			"steps": 35,
			"cfg": 7,
			"sampler_name": "dpmpp_3m_sde_gpu",
			"scheduler": "exponential",
			"denoise": 1,
			"model": ["5",0],
			"positive": ["8",0],
			"negative": ["8",1],
			"latent_image": ["12",0]
		},
		"class_type": "KSampler",
		"_meta": {"title": "KSampler"}
	},
	"12": {
		"inputs": {"width": 1024,"height": 1024,"batch_size": 1},
		"class_type": "EmptyLatentImage",
		"_meta": {"title": "Empty Latent Image"}
	},
	"15": {
		"inputs": {"samples": ["11",0], "vae": ["5",2]},
		"class_type": "VAEDecode",
		"_meta": {"title": "VAE Decode"}
	},
	"16": {
		"inputs": {"filename_prefix": "ComfyUI", "images": ["15",0]},
		"class_type": "SaveImage",
		"_meta": {"title": "Save Image"}
	}
}

/*
	===============================================
	OPENAI DALLE GENERATOR
	===============================================
*/

setup.evaluateDalleCharacterDescription = function(mc) {
	let description = ``;
	if (mc.sex === "male") {
		description += "The character is a male. ";
	} else if (mc.sex === "female") {
		description += "The character is a female. ";
	} else {
		description += "";
	};

	description += `${mc.hair} colored hair. `;
	description += `${mc.eyeColor} colored eyes. `;
	description += `${mc.skinType} ${mc.skinColor} colored skin. `;

	if (mc.ears != "normal human") description += `${mc.ears} ears. `;

	if (mc.appAge < 15) {
		description += `A child. `;
	} else if (mc.appAge < 20) {
		description += `A teenager. `;
	} else if (mc.appAge < 30) {
		description += `A young adult. `;
	} else if (mc.appAge < 45) {
		description += `An adult. `;
	} else if (mc.appAge < 55) {
		description += `A middle-aged adult. `;
	} else if (mc.appAge < 65) {
		description += `And older adult. `;
	} else {
		description += `And elderly adult. `;
	}

	if (mc.subdom > 0) {
		description += "with a very shy body posture. ";
	} else if (mc.subdom < 0) {
		description += "with a very strong body posture. ";
	}

	if (mc.hasCurse("Horny")) {
		// Ensure these variables are defined
		let hornCount = State.variables.hornCount || 0;
		let hornAdjective = State.variables.hornAdjective || "";
		let hornVariation = State.variables.hornVariation || "";
		description += `with ${(hornCount === 1) ? "a" : "two"} noticeable ${hornAdjective} ${hornVariation} horn${(hornCount > 1) ? "s" : ""}. `;
	}

	// Gender and physical appearance
	switch(mc.gender) {
		case 1: description += "A masculine man. "; break;
		case 2: description += "A feminine man (twink, femboy). "; break;
		case 3: description += "A very androgynous man. "; break;
		case 4: description += "A very androgynous woman. "; break;
		case 5: description += "A masculine woman (tomboy). "; break;
		case 6: description += "A feminine woman. "; break;
	}

	// Breast size
	if (mc.breastsCor < 1 && mc.vagina === 0) {
		description += "";
	} else if (mc.breastsCor < 1 && mc.vagina === 1) {
		description += "with a totally flat chest. ";
	} else if (mc.breastsCor < 6) {
		description += "";
	} else {
		description += "with an abnormally large chest. ";
	}

	// Additional conditions
	if (mc.dollevent2) description += "Wearing a tattered pink dress, resembling a child's doll. ";

	// Pregnancy
	const pregnantDays = setup.daysConsideredPregnant(mc);
	if (120 <= pregnantDays && pregnantDays < 180) {
		description += mc.menFirstCycle ? "A noticeable pregnancy bump. " : "A small pregnancy bump. ";
	} else if (180 <= pregnantDays && pregnantDays < 240) {
		description += "A large pregnancy bump. ";
	} else if (pregnantDays >= 240 && setup.daysUntilDue(mc) > 0) {
		description += "A huge pregnancy belly. ";
	}

	// Curses and conditions
	if (mc.hasCurse("Freckle Speckle")) description += "Many freckles. ";
	if (mc.hasCurse("20/20000000")) description += "Wearing thick glasses. ";
	if (mc.hasCurse("Gooey")) description += "A slime person, transluscent slime-skinned. ";
	if (mc.hasCurse("Crossdress Your Heart")) description += "Crossdressing as the opposite gender. ";
	if (mc.hasCurse("Lingual Leviathan")) description += "A very long tongue sticking out. ";
	if (mc.hasCurse("Massacre Manicure")) description += "Abnormally sharp and long fingernails. ";
	if (mc.hasCurse("Flower Power")) description += "Covered in flowers. ";
	if (mc.hasCurse("Cellulose")) description += "Made of living plant matter, like a dryad. ";
	if (mc.hasCurse("Wriggly Antennae")) description += "Wriggly insect antennae on forehead. ";
	if (mc.hasCurse("Carapacian")) description += "Covered in an insect-like carapac. ";
	if (mc.hasCurse("Creature of the Night")) description += "A vampire, with vampire fangs. ";
	if (mc.hasCurse("Minish-ish")) description += `Very tiny, only a few inches tall. `;
	if (mc.hasCurse("Colossal-able")) description += `Enormous, asbolutely giant. `;
	if (mc.hasCurse("Seafolk")) description += "A merfolk with a merfolk tail. ";
	if (mc.hasCurse("Tickly Tentacles")) description += `${mc.tentacles} squirming tentacles growing from their body. `;
	if (mc.hasCurse("Eye-scream")) description += `${mc.extraEyes} extra eyes on their body. `;
	if (mc.hasCurse("A Mouthful")) description += `${mc.extraMouths} extra mouths on their body. `;
	if (mc.hasCurse("Below the Veil")) description += "A strange, eldritch entity that seems very creepy and *wrong* in subtle ways. ";
	description += "\n"+ SugarCube.setup.manualDesc;

	return description;
};

setup.openAI_InvokeDalleGenerator = async function(prompt) {
	const apiKey = settings.OpenAIAPIKey;

	const headers = {
		'Content-Type': 'application/json',
		'Authorization': `Bearer ${apiKey}`
	}

	const body = JSON.stringify({
		model: 'dall-e-3',
		prompt: prompt,
		n: 1,
		size: "1024x1024",
		response_format: "b64_json"
	})

	const response = await fetch('https://api.openai.com/v1/images/generations', {
		method: 'POST',
		headers: headers,
		body: body
	});

	// check if we connected to OpenAI
	if (!response.ok) {
		is_generation_busy = false;
		throw new Error('Failed to connect to OpenAI. Please check your API key and network connection and try again. If those are both correct, this may be due to a content policy error from OpenAI.');
	}

	// check if any image was given from Dalle
	const data = await response.json();
	// console.log(data);

	if (!data.data || data.data.length == 0) {
		is_generation_busy = false;
		console.error('No images returned from Dalle:', data);
		throw new Error('No images returned from server. This is likely due to a content policy error or server error from OpenAI.');
	}

	// once we receive the image, save it as the player portrait
	const storeKey = "dalleImage";
	const b64Image = data.data[0].b64_json;
	// console.log("Base64 Data Length: ", b64Image.length);

	// now save it in the local storage
	try {
		setup.storeImage(storeKey, b64Image);
		// console.log('Image successfully stored.');
	} catch(error) {
		console.error('Failed to store image due to error:', error);
	}
}

setup.openAI_GenerateDallePortrait = async function() {
	if (is_generation_busy) {
		return;
	}
	is_generation_busy = true;

	// Notification element
	const notificationElement = document.getElementById('notification');

	// Static part of the prompt
	let staticPrompt = "Create an anime-inspired digital painting of a single character with each of the following traits. You must keep in mind every physical trait below. You must use an *anime-inspired digital painting* style. The character is an adventurer and the background of the scene is the Abyss from MiA. Do NOT use the word character in the final prompt.\n\nCharacter traits:\n";

	// Dynamically generated character description
	let characterDescription = setup.evaluateDalleCharacterDescription(State.variables.mc); // Assuming $mc is stored in State.variables.mc

	// Concatenate the static prompt with the dynamic description
	var prompt = staticPrompt + characterDescription;

	if (setup.customPromptPrefix != null) {
		prompt = setup.customPromptPrefix + "," + prompt
	}

	if (setup.customPromptSuffix != null) {
		prompt = prompt + "," + setup.customPromptSuffix
	}

	try {
		await setup.openAI_InvokeDalleGenerator(prompt);
		if (notificationElement != null) {
			notificationElement.style.display = 'hidden';
		}
	} catch (error) {
		console.error('Error generating image:', error);
		if (notificationElement != null) {
			notificationElement.style.display = 'block';
			notificationElement.textContent = 'Error generating image: ' + error.message + (error.response ? (await error.response.json()).error : ' No additional error information from OpenAI.');
		}
	}

	is_generation_busy = false;
}

/*
	===============================================
	(LOCAL) COMFYUI GENERATOR
	===============================================
*/

setup.DEFAULT_CHECKPOINT = "hassakuXLPony_v13BetterEyesVersion";
setup.DEFAULT_LORAS = ["DallE3-magik"]
setup.DEFAULT_POSITIVE_TAGGING = ["score_9_up", "score_8_up", "score_7_up", "cowboy shot", "1girl", "solo", "source_anime", "front view"]
setup.DEFAULT_NEGATIVE_TAGGING = ["score_5_up", "score_4_up", "score_3_up", "pony", "ugly", "ugly face", "poorly drawn face", "blurry", "blurry face", "(3d)", "realistic", "muscular", "long torso", "blurry eyes", "poorly drawn eyes", "patreon", "artist name", "sd", "super deformed"]

// https://civitai.com/articles/5473/pony-cheatsheet-v2

// https://civitai.com/articles/6349/280-pony-diffusion-xl-recognized-clothing-list-booru-tags-sfw
// https://civitai.com/articles/6888/320-pony-diffusion-xl-character-hairstyles-ears-wings-and-tails-booru-tags-sfw
// https://civitai.com/articles/7579/480-pony-diffusion-xl-hats-masks-and-more-props-list-booru-tags-sfw

setup.AVAILABLE_MODELS = ["hassakuXLPony_v13BetterEyesVersion"];
setup.AVAILABLE_LORAS = ["DallE3-magik"];

setup.POSITIVE_CATEGORICAL_TAGGING = {
	"Pony Scores" : [
		"score_9_up", "score_8_up", "score_7_up", "score_6_up", "score_5_up", "score_4_up"
	],

	"Image Quality" : [
		"best_quality", "masterpiece", "hd", "4k", "8k", "extremely detailed", "inctricate details", "ultra-detailed",
		"illustration", "detailed light", "hdr", "best quality", "amazing quality", "very aesthetic", "absurdres",
		"extremely detailed CG unity 8k wallpaper", "highres", "highly detailed", "soft shadow", "hard shadow",
		"strong shadow", "depth of field", "an extremely delicate and beautiful"
	],

	"Pony Source" : [
		"source_anime", "source_cartoon", "source_pony", "source_fluffy"
	],

	"Ratings" : [
		"rating_explicit", "rating_questionable", "rating_safe"
	],

	"Camera Angles" : [
		"top down", "birds eye view", "high angleshot", "above shot", "slightly above", "straight on", "front view",
		"hero view", "cowboy shot", "low view", "worms eye view"
	],

	"Camera Location" : [
		"extreme long shot", "long shot", "medium long shot", "medium shot", "medium close up", "close up",
		"extreme close up", "facing towards viewer", "facing away from viewer"
	],

	"Quantity" : [
		"solo", "1girl", "2girls", "3girls", "4girls", "1 girl", "2 girls", "3 girls", "4 girls",
		"multiple_girls", "1boy", "2boys", "3boys", "4boys", "1 boy", "2 boys", "3 boys", "4 boys",
		"multiple_boys"
	],

	"NSFW" : [
		"bra", "panties", "nude", "naked"
	],

	"Censor" : [
		"censored", "mosiac censorship", "out-of-frame censoring", "bar censor",
		"heart censor", "glitch censor", "shadow censor", "tape censor", "blue censor", "tail censor", "ribbon censor",
		"smoke censor", "feather censor", "soap censor", "one finger selfie challenge", "light censor"
	],

	"Curses" : [
		// TODO: hook to game curses -> in the prompt code it will output the curse prompt -> append that to the custom tagging
	],
}

setup.NEGATIVE_CATEGORICAL_TAGGING = {
	"Pony Scores" : ["score_3_up", "score_4_up", "score_5_up"],

	"Image Quality" : [
		"flat color", "lowres", "bad hands", "bad fingers", "missing fingers", "extra digit", "fewer digits",
		"worst quality", "normal quality", "jpeg artifacts", "signature", "watermark", "username", "bad feet",
		"three legs", "wrong hand", "wrong feet", "wrong fingers", "deformed leg", "abnormal", "malformed",
		"bad art", "deformed", "disfigured", "mutation", "mutated", "extra limbs", "inaccurate limb",
		"missing limb", "floating limbs", "disconnected limbs", "(pencil neck)", "long neck", "long body", "mutated skeleton",
		"long skeleton", "bad proportions", "mutated hands and fingers", "poorly drawn hands", "malformed hands",
		"poorly drawn face", "poorly drawn asymmetrical eyes", "mutated face", "low quality", "distorted light",
		"low quality illustration", "blurry", "bad anatomy"
	]
}

setup.comfyUI_ClearAdvanced = function() {
	SugarCube.State.variables.advancedMenuCheckpoint = setup.DEFAULT_CHECKPOINT;
	SugarCube.State.variables.advancedMenuLORAs = new Set();
	SugarCube.State.variables.advancedMenuPositive = new Set();
	SugarCube.State.variables.advancedMenuNegative = new Set();
}

setup.comfyUI_GenerateAdvancedParameters = function() {
	var checkpoint = SugarCube.State.variables.advancedMenuCheckpoint;
	var steps = 20;
	var cfg = 7.0;
	var width = 1024;
	var height = 1024;

	var seed = null;
	if (setup.customSeedInput == null || setup.customSeedInput == "" || !Number.isInteger(parseInt(setup.customSeedInput))) {
		seed = SugarCube.State.prng.seed | Math.round(Math.random() * 10_000);
	} else {
		seed = parseInt(setup.customSeedInput);
	}

	// positive
	let positive = "";

	for (let [tag, value] of Object.entries(SugarCube.State.variables.advancedMenuLORAs)) {
		if (value) {
			let lora_tag = "<lora:" + tag + ":1> ";
			positive += lora_tag;
		}
	}

	for (let [tag, value] of Object.entries(SugarCube.State.variables.advancedMenuPositive)) {
		if (value) {
			positive += tag + ",";
		}
	}

	// negative
	let negative = "";
	for (let [tag, value] of Object.entries(SugarCube.State.variables.advancedMenuNegative)) {
		if (value) {
			negative += tag + ",";
		}
	}

	// if curses is enabled, append that
	if (SugarCube.State.variables.UseMixedGameComfyUIPrompt == true) {
		const [curses_positive, curses_negative] = setup.comfyUI_GenerateCurseParameters();
		positive += "," + curses_positive
		negative += "," + curses_negative
	}

	return [positive, negative, checkpoint, steps, cfg, seed, width, height]
}

setup.comfyUI_ResetAdvanced = function() {
	// set default model
	SugarCube.State.variables.advancedMenuCheckpoint = setup.DEFAULT_CHECKPOINT;

	// set default loras
	SugarCube.State.variables.advancedMenuLORAs = new Set();
	for (let tag of Object.values(setup.DEFAULT_LORAS)) {
		SugarCube.State.variables.advancedMenuLORAs.add(tag);
	}

	SugarCube.State.variables.advancedMenuPositive = new Set();
	for (let tag of Object.values(setup.DEFAULT_POSITIVE_TAGGING)) {
		SugarCube.State.variables.advancedMenuPositive.add(tag);
	}

	SugarCube.State.variables.advancedMenuNegative = new Set();
	for (let tag of Object.values(setup.DEFAULT_NEGATIVE_TAGGING)) {
		SugarCube.State.variables.advancedMenuNegative.add(tag);
	}
}

setup.ComfyUI_GenerateAdvancedHTMLPage = function() {
	var html_text = "";

	// if it hasn't been set yet, set them all to the defaults
	if (SugarCube.State.variables.advancedMenuCheckpoint == "" || SugarCube.State.variables.advancedMenuCheckpoint == null) {
		setup.comfyUI_ResetAdvanced();
	}

	// checkpoints
	html_text += "<h2>Checkpoints</h2>";
	for (let tag of Object.values(setup.AVAILABLE_MODELS)) {
		let current_value = (SugarCube.State.variables.advancedMenuCheckpoint == tag);
		html_text += tag + ": <<checkbox \"$advancedMenuCheckpoint\" null " + tag + " " + (current_value ? "checked" : "autocheck") + ">> ";
	}
	html_text += "\n";

	// loras
	html_text += "<h2>LORAs</h2>";
	for (let tag of Object.values(setup.AVAILABLE_LORAS)) {
		let current_value = SugarCube.State.variables.advancedMenuLORAs.has(tag) == true;
		html_text += tag + ": <<checkbox \"$advancedMenuLORAs[\'" + tag + "\']\" false true " + (current_value ? "checked" : "autocheck") + ">> ";
	}
	html_text += "\n";

	// positive prompt
	html_text += "<h2>Positive Prompt</h2>";
	for (let [category, array_of_tags] of Object.entries(setup.POSITIVE_CATEGORICAL_TAGGING)) {
		html_text += "<h4>" + category + "</h4>";
		for (let tag of Object.values(array_of_tags)) {
			let current_value = SugarCube.State.variables.advancedMenuPositive.has(tag) == true;
			html_text += tag + ": <<checkbox \"$advancedMenuPositive[\'" + tag + "\']\" false true " + (current_value ? "checked" : "autocheck") + ">> ";
		}
		html_text += "\n";
	}
	html_text += "\n"

	// negative prompt
	html_text += "<h2>Negative Prompt</h2>";
	for (let [category, array_of_tags] of Object.entries(setup.NEGATIVE_CATEGORICAL_TAGGING)) {
		html_text += "<h4>" + category + "</h4>";
		for (let tag of Object.values(array_of_tags)) {
			let current_value = SugarCube.State.variables.advancedMenuNegative.has(tag) == true;
			html_text += tag + ": <<checkbox \"$advancedMenuNegative[\'" + tag + "\']\" false true " + (current_value ? "checked" : "autocheck") + ">> ";
		}
		html_text += "\n";
	}

	// curses

	return html_text;
}

setup.updateComfyUIStatus = async function() {
	const url = "http://127.0.0.1:8188";

	var is_running = false;

	try {
		await fetch(url, {method: 'GET', headers: {'Origin' : 'AbyssDiver.html', 'Content-Type': 'application/json'}});
		is_running = true;
		// try display any image if any are available
		// setup.displaySavedImage().catch(() => null)
		setup.displayRecentGeneratedImage().catch(() => null)
	} catch (error) {}

	const notificationElement = document.getElementById('comfyui-enabled');
	notificationElement.textContent = is_running ? "ComfyUI is currently running." : "ComfyUI is NOT currently running."
	notificationElement.style.color = is_running ? "green" : "red"
}

setup.comfyUI_InvokeGenerator = async function(url, payload) {
	// console.log(url, JSON.stringify(payload));

	var response = null;
	try {
		response = await fetch(url, {
			method: 'POST',
			headers: {'Origin' : 'AbyssDiver.html', 'Content-Type': 'application/json'},
			body: JSON.stringify(payload)
		});
	} catch (error) {
		throw new Error('Failed to connect to Proxy. Please check your Proxy and ensure the server is running.');
	}

	if (!response.ok) {
		throw new Error('Failed to connect to Proxy. Please check your Proxy and ensure the server is running.');
	}

	const data = await response.json();
	// Debugging: Inspect the structure of the response
	// console.log(data);
	// Return the data for further processing
	return data;
}

function processClothingState(mc_state, mc_curses) {
	const thresholds = [0, 4, 6, 8, 10];

	const hasCurse = (curseName) => {
		return mc_curses.some(curse => curse.constructor.name === curseName);
	};

	const getHighestReachedThreshold = (value) => {
		for (let i = thresholds.length - 1; i >= 0; i--) {
			if (value >= thresholds[i]) {
				return thresholds[i];
			}
		}
		return 0;
	};

	const currentThreshold = getHighestReachedThreshold(mc_state.libido);
	const hasCRB = hasCurse('ClothingRestrictionB');
	const hasCRC = hasCurse('ClothingRestrictionC');

	const processState = () => {
		switch (true) {

			case !hasCRB && !hasCRC:
				switch (currentThreshold) {
					case 0:
						// console.log('Initial state - No restrictions');
						return["fully clothed, modest","modestClothing"];
					case 4:
						// console.log('Clothing passed 4 - No restrictions');
						return["normal","normalClothing"];
					case 6:
						// console.log('Clothing passed 6 - No restrictions');
						return["immodest","immodestClothing"];
					case 8:
						// console.log('Clothing passed 8 - No restrictions');
						return["skimpy","skimpyClothing"];
					case 10:
						// console.log('Clothing passed 10 - No restrictions');
						return["slutty","sluttyClothing"];
				}
				break;
			case hasCRB && hasCRC:
				switch (currentThreshold) {
					case 0:
						// console.log('Initial state - Both B and C restrictions active');
						return["no","nude"];
					case 4:
						// console.log('Clothing passed 4 - Both B and C restrictions active');
						return["no","nude"];
					case 6:
						// console.log('Clothing passed 6 - Both B and C restrictions active');
						return["no","nude"];
					case 8:
						// console.log('Clothing passed 8 - Both B and C restrictions active');
						return["no","nude"];
					case 10:
						// console.log('Clothing passed 10 - Both B and C restrictions active');
						return["no","nude"];
				}
				break;
			case hasCRB && !hasCRC:
				switch (currentThreshold) {
					case 0:
						// console.log('Initial state - ClothingRestrictionB active');
						return["fully clothed, modest","modestClothing"];
					case 4:
						// console.log('Clothing passed 4 - ClothingRestrictionB active');
						return["normal","normalClothing"];
					case 6:
	 					// console.log('Clothing passed 6 - ClothingRestrictionB active');
						return["skimpy","skimpyClothing"];
					case 8:
						// console.log('Clothing passed 8 - ClothingRestrictionB active');
						return["slutty","sluttyClothing"];
					case 10:
						// console.log('Clothing passed 10 - ClothingRestrictionB active');
						return["slutty","sluttyClothing"];
				}
				break;

			case !hasCRB && hasCRC:
				switch (currentThreshold) {
					case 0:
						// console.log('Initial state - ClothingRestrictionC active');
						return["skimpy","skimpyClothing"];
					case 4:
						// console.log('Clothing passed 4 - ClothingRestrictionC active');
						return["slutty","sluttyClothing"];
					case 6:
						// console.log('Clothing passed 6 - ClothingRestrictionC active');
						return["slutty","sluttyClothing"];
					case 8:
						// console.log('Clothing passed 8 - ClothingRestrictionC active');
						return["slutty","sluttyClothing"];
					case 10:
						// console.log('Clothing passed 10 - ClothingRestrictionC active');
						return["slutty","sluttyClothing"];
				}
				break;
		}
	};

	// Return the processed state
	return processState();
}

function processRanges(ranges) {
	const result = {};

	for (const [rangeName, [value, thresholds]] of Object.entries(ranges)) {
		const sortedThresholds = [...thresholds].sort((a, b) => a[1] - b[1]);	
		const matchedThreshold = sortedThresholds.reduce((acc, [description, threshold]) => {
			return value >= threshold ? { description, threshold } : acc;
		}, sortedThresholds[0]);
		result[rangeName] = matchedThreshold.description;
	}

	return result;
}

setup.comfyUI_PrepareCharacterData = function() {
	const mc_booleans = {
		//clothing
		modestClothing: false,
		normalClothing: false,
		immodestClothing: false,
		skimpyClothing: false,
		sluttyClothing: false,
		naked: false,
		nude: false,
		//items
		moonwatcher: false,
		//expression
		closedMouth: false,

	};
	// console.log(mc_booleans);

	// get the character curses
	const mc_curses = State.variables.mc.curses;

	const mc_state = {
		'age' : State.variables.mc.appAge, //years, final calculated age
		'gender' : State.variables.mc.gender, //0 guy, 3.5 androgynous, 7 girl
		// 'apparent_gender' : State.variables.mc.appGender, //counts things like smell and voice, much worse metric to use 
		'penis' : State.variables.mc.penisCor, //0 means none,  measured in inches
		// 'vagina_count' : State.variables.mc.vagina, //avoiding this for now, not enough prompt testing
		// 'double_penis' : State.variables.mc.doublePenis, //avoiding this for now, not enough prompt testing
		'sex' : State.variables.mc.sex, // female, male, futa
		'lactation' : State.variables.mc.lactation, //0 nothing, > 0 some lactation, > 1 extreme lactation
		'breasts' : State.variables.mc.breastsCor, //Corrected size, base 0, 
		// 'breastsLabel' : State.variables.mc.breastsLabel, //Letter Size, not used
		'height' : State.variables.mc.heightCor, //height in cms
		'libido' : State.variables.mc.libido, //base 2, higher means higher libido
		'subdom' : State.variables.mc.subdom, //base 0, positive 2 most submissive, -1 dominant
		'hair' : State.variables.mc.hair, //colour
		'ears' : State.variables.mc.ears, //type, default "normal human"
		'bodyHair' : State.variables.mc.bodyHair, //0 meaning none, 1 being normal, 2 being maximum fluff 
		// 'skinType' : State.variables.mc.skinType, // not usable for prompt
		'skinColor' : State.variables.mc.skinColor, // is adjusted by curses, use directly
		'eyeColor' : State.variables.mc.eyeColor, // is adjusted by curses, use directly
		'tail' : State.variables.mc.tail[0], //mc.tail is an ARRAY, you can't just reference it directly. Changed to tail[0]
		// 'description' : State.variables.mc.desc, //I have NO idea what this does
		'blood' : State.variables.mc.blood, //blood colour 
		// 'genderVoice' : State.variables.mc.genderVoice, //Voices aren't used in the prompt
		// 'fluids' : State.variables.mc.fluids, //100 is normal, 0 is none, 200 is double, not used for portraits
		// 'lewdness' : State.variables.mc.lewdness, //Only libido will be referenced, this is too vague to combine the two
		'horns' : State.variables.mc.horns, // Horn count
		'hornType' : State.variables.hornVariation, // Horn type, not stored in the MC for some reason??
		// 'inhuman' : State.variables.mc.inhuman, // not used for portraits
		// 'eyeCount' : State.variables.mc.eyeCount, // Not using for now, too intrusive on prompt
		// 'armCount' : State.variables.mc.armCount, // Not using for now, too intrusive on prompt
		// 'legCount' : State.variables.mc.legCount, // Not using for now, too intrusive on prompt
		// 'tentacles' : State.variables.mc.tentacles,
		// 'extraEyes' : State.variables.mc.extraEyes, // Not using for now, too intrusive on prompt
		// 'extraMouths' : State.variables.mc.extraMouths // Not using for now, too intrusive on prompt
		'fitness' : State.variables.mc.fit, //-2 is out of shape, 0 is normal, 2 is very fit
	};
	// console.log(mc_state);

	const mc_ranges = {
		"sex": [mc_state.gender, [["1guy", 0],["1girl", 3.5]]],
		"height": [mc_state.height, [["Pixie Sized", 0], ["extremely short", 120], ["very short", 140], ["short", 150], ["below average height", 160], ["average height", 170], ["above average height", 180], ["tall", 190], ["very tall", 200], ["extremely tall", 210], ["titanic height", 300]]],
		"penis": [mc_state.penis, [["", 0],["micro penis", 0.1],["tiny penis", 2],["small penis", 3], ["below average penis", 4], ["medium penis", 5], ["above average penis", 7], ["large penis", 9], ["huge penis", 11], ["gigantic penis", 13], ["hyper penis", 20]]],
		"clothing": [mc_state.libido, [["modestly dressed, fully clothed", 0], ["normal clothing" , 4], ["immodest clothing", 6], ["skimpy clothing", 8], ["slutty slothing", 10]]],
		"fitness": [mc_state.fitness, [["fragile body", -2], ["weak body", -1], ["average fitness", 0], ["fit body", 1], ["very fit body", 2]]],
		"breasts": [mc_state.breasts, [[`${mc_state.gender < 4 ? "male body:1.2" : "flat chest:1.3"}`, 0], ["flat breasts:1.3", 1], ["small breasts:1.2", 3], ["medium breasts:1.2", 4], ["large breasts", 6], ["huge breasts", 8], ["gigantic breasts", 10], ["hyper breasts", 20]]],
		"gender": [mc_state.gender, [["hyper masculine, man", 0], ["masculine, man", 1], ["twink", 2], ["femboy", 3], ["tomboy", 4], ["androgynous woman", 5], ["feminine, woman", 6], ["hyper feminine, woman", 7]]],
		"age": [mc_state.age, [["(elder:3) (old:3)", 0], ["young adult", 14], ["adult", 25], [`adult, older, ${mc_state.gender < 4 ? "Dilf" : "Milf"}`, 35], [`(old ${mc_state.gender < 4 ? "man" : "woman"})`, 50], ["(old:1.5), senior citizen", 60], ["(old:2), senior citizen", 70]]],
		"subdom": [mc_state.subdom, [["dominant", -2], ["confident", -1], ["relaxed", 0], ["timid", 1], ["submissive", 2]]],
		"lactation": [mc_state.lactation, [["", 0], [", lactation", 1], [", (lactation:1.3)", 2]]],
	};

	// console.log(mc_ranges);
	const mc_range = processRanges(mc_ranges);
	// console.log(mc_range);

	let [clothingString, clothingTrue] = processClothingState(mc_state, mc_curses);
	mc_booleans[clothingTrue] = true;
	mc_state.clothing = clothingString;
	// console.log(mc_state.clothing+" "+mc_booleans.modestClothing);

	// payload to send to proxy/comfyui
	const payload = {'booleans' : mc_booleans, 'curses' : mc_curses, 'state' : mc_state, 'range' : mc_range};
	return payload;
}

setup.comfyUI_GenerateStatParameters = function(characterData) {
	const state = characterData.state;
	const range = characterData.range;
	characterData.statPrompts = {
		"misprints":{
			positive:[],
			negative:["artist name","patreon", "text", "watermark", "jpeg artifacts", "signature", "username"],
			excludedBy:[]
		},
		"quality":{
			positive:["score_9", "score_8_up", "score_7_up"],
			negative:["easynegative", "score_5_up", "score_4_up", "score_3_up", "blurry", "sd", "lowres", "worst quality", "bad art"],
			excludedBy:[]
		},
		"framing":{
			positive:["(medium close up:1.3)", "solo", `${range.sex}`],
			negative:["full shot", "(full body:0.3)", "(cowboy shot:0.3)", "(medium full shot:0.1)", "(hand on another's head:1.5)"],
			excludedBy:[]
		},
		"background":{
			positive:["green background", "blank background", "no foreground"],
			negative:["distorted light"],
			excludedBy:[]
		},
		"style":{
			positive:["source anime", "(human body:0.5)"],
			negative:["(3d)", "realistic", "super deformed", "(pony:1.5)", "ugly", "flat color", "abnormal", "malformed", "disfigured", "bad anatomy", "censored", "blurry"],
			excludedBy:[]
		},
		"age":{
			positive:[`(${range.age}:1.4)`],
			negative:["(toddler:1.3)", "(kindergartener:1.3)", "(child:1.3)", "(underage:1.3)", "early teen", "(shota:1.3)", "(loli:1.3)"],
			excludedBy:[]
		},
		"gender":{
			positive:[`${range.gender}`],
			negative:[],
			excludedBy:[]
		},
		"fitness":{
			positive:[],
			negative:[],
			excludedBy:[]
		},
		"libido":{
			positive:[],
			negative:[],
			excludedBy:[]
		},
		"behaviour":{
			positive:[`${range.subdom} attitude`, "standing straight"],
			negative:[],
			excludedBy:[]
		},
		"breasts":{
			positive:[`(${range.breasts})`],
			negative:[],
			excludedBy:[]
		},
		"nipples":{
			positive:[`normal nipples${range.lactation}`],
			negative:[],
			excludedBy:[]
		},
		"clothing":{
			positive:[`(${state.clothing} clothing:1.2)`],
			negative:[],
			excludedBy:[]
		},
		"ears":{
			positive:[`(${state.ears} ears:1.2)`],
			negative:["(animal ears:1.2)", "(four ears:1.2)", "(extra ears:1.2)", "pointed ears", "elf ears"],
			excludedBy:[]
		},
		"eyes":{
			positive:[`${state.eyeColor} eyes`],
			negative:["(blurry eyes:1.3)", "(poorly drawn eyes:1.3)", "(asymmetrical eyes:1.3)"],
			excludedBy:[]
		},
		"face":{
			positive:[],
			negative:["blurry face", "ugly face", "poorly drawn face", "(pencil neck)", "long neck", "mutated face"],
			excludedBy:[]
		},
		"hair":{
			positive:[`(${state.hair} hair:1.2)`],
			negative:[],
			excludedBy:[]
		},
		"limbs":{
			positive:["(human hands:0.5)", "(human fingers:0.5)"],
			negative:["extra limbs", "inaccurate limb", "bad hands", "bad fingers", "missing fingers", "extra digit", "fewer digits", "wrong hand", "wrong fingers", "floating limbs", "disconnected limbs", "paws", "animal hands"],
			excludedBy:[]
		},
		"mouth":{
			positive:[],
			negative:[],
			excludedBy:[]
		},
		"penis":{
			positive:[`${range.penis}`],
			negative:[],
			excludedBy:[!range.penis]
		},
		"pregnancy":{
			positive:[],
			negative:[],
			excludedBy:[]
		},
		"skin":{
			positive:[`(${state.skinColor} skin:1.2)`],
			negative:[],
			excludedBy:[]
		},
		"tail":{
			positive:[`${state.tail} tail`],
			negative:["Multiple tails", "2 tails", "extra tails"],
			excludedBy:[!(state.tail)]
		},
		"torso":{
			positive:[`${range.height}`, `${range.fitness}`],
			negative:["animal body","(floating body)", "(floating torso:1.5)","long torso", "mutated skeleton", "long skeleton", "bad proportions"],
			excludedBy:[]
		},
		"vulva":{
			positive:[],
			negative:[],
			excludedBy:[]
		},
	};
	//console.log(characterData.statPrompts);
	Object.entries(characterData.statPrompts || {})
		.forEach(([category, categoryData]) => {
			characterData.statPromptArray = characterData.statPromptArray || {};
			characterData.excluded = characterData.excluded || {};

			characterData.statPromptArray[category] = characterData.statPromptArray[category] || { positive: [], negative: [] };
			characterData.excluded[category] = characterData.excluded[category] || { positive: [], negative: [] };

			const excludedReasons = (categoryData.excludedBy || [])
				.map(condition => condition ? condition : null).filter(Boolean);

			['positive', 'negative'].forEach(type => {
				(categoryData[type] || []).forEach(tag => {
					excludedReasons.length === 0
					? characterData.statPromptArray[category][type].push(tag)
					: characterData.excluded[category][type].push({
						tag: tag
					});
				});
			});
		});
		// console.log(characterData.statPromptArray);
	return characterData;
}

setup.comfyUI_GenerateCurseParameters = function(characterData) {
	const hasCurse = (curseName) => {
		return characterData.curses.some(curse => curse.constructor.name === curseName);
	};
	const state = characterData.state;
	const range = characterData.range;
	const bool = characterData.booleans;
	characterData.cursePrompts ={
		"ClothingRestrictionA": {
			"affects" : {
				"clothing":{
					"positive": ["no accessories"],
					"negative": ["scarves", "hats", "earrings", "piercings", "pasties", "jewellery", "rings", "necklaces", "gloves", "bracelets", "hair clips", "hair ornaments"],
					"excludedBy": [],
				}
			}
		},
		"HairRemoval": {
			"affects": {
				"skin": {
					"positive": ["waxed body", "smooth skin"],
					"negative": ["(female pubic hair:1.5)","(male pubic hair:1.5)", "(facial hair:1.5)"],
					"excludedBy": [hasCurse("MaximumFluff")],
				},
				"eyes": {
					"positive": ["trimmed eyebrows"],
					"negative": [],
					"excludedBy": [],
				}
			}
		},
		"FreckleSpeckle": {
				"affects": {
					"skin": {
						"positive": ["(freckles)","(body freckles:1.1)", "(freckles on face:1.1)", "(freckles on chest:1.1)", "(freckles on arms:1.1)"],
						"negative": [],
						"excludedBy": [],
					}
			}
		},
		"KnifeEar": {
			"affects": {
				"ears": {
					"positive": [`(pointed ears:1.2)`],
					"negative": [],
					"excludedBy": [],
				}
			}
		},
		// "IncreasedSensitivity": {
		// 	"affects": {
		// 		"behaviour": {
		// 			"positive": [],
		// 			"negative": [],
		// 			"excludedBy": [],
		// 		}
		// 	}
		// },
		"FluffyEars": {
			"affects": {
				"ears": {
					"positive": ["(animal ears:1.2)", "(animal ears only)"],
					"negative": ["(human ears:1.5)", "(normal human ears:1.2)", "4 ears"],
					"excludedBy": [],
				},
				"hair": {
					"positive": [`${state.ears} ears in hair`],
					"negative": [],
					"excludedBy": [],
				}
			}
		},
		"FluffyTail": {
			"affects": {
				"tail": {
					"positive": [`${state.hair} tail fur`],
					"negative": ["multiple tails"],
					"excludedBy": [],
				}
			}
		},
		"MaximumFluff": {
			"affects": {
				"skin": {
					"positive": [`(fluffy ${state.hair} fur cover body:1.2)`],
					"negative": [],
					"excludedBy": [],
				},
				"hair": {
					"positive": ["fur like hair on head"],
					"negative": [],
					"excludedBy": [],
				},
				"limbs": {
					"positive": [`fluffy ${state.hair} arms`],
					"negative": [],
					"excludedBy": [],
				}
			}
		},
		"HeatRut": {
			"affects": {
				"behaviour": {
					"positive": ["(heavy_breathing:1.4)","(full-face blush:1.2)"],
					"negative": [],
					"excludedBy": [],
				},
				"breast":{
					"positive": ["(chest blush:1.3)"],
					"negative": [],
					"excludedBy": [bool.modestClothing, bool.normalClothing],
				},
				"pregnancy": {
					"positive": [],
					"negative": [],
					"excludedBy": [],
				}
			}
		},
		// "Lightweight": {
		// 	"affects": {
		// 		"behaviour": {
		// 			"positive": [],
		// 			"negative": [],
		// 			"excludedBy": [],
		// 		}
		// 	}
		// },
		"BlushingVirgin": {
			"affects": {
				"behaviour": {
					"positive": ["shy"],
					"negative": [],
					"excludedBy": [],
				}
			}
			},
		// "SubmissivenessRectificationA": {
		// 	"affects": {
		// 		"behaviour": {
		// 			"positive": ["timid"],
		// 			"negative": [],
		// 			"excludedBy": [],
		// 		}
		// 	}
		// },
		"ClothingRestrictionB": {
			"affects": {
				"clothing": {
					"positive": ["No undergarments"],
					"negative": ["underwear", "boxers", "briefs", "panties", "thongs", "socks", "stockings", "bra"],
					"excludedBy": [bool.modestClothing, bool.normalClothing],
				}
			}
		},
		"PowerDom": {
			"affects": {
		// 		"behaviour": {
		// 			"positive": ["confident", "dominatrix"],
		// 			"negative": [],
		// 			"excludedBy": [],
		// 		}
				"clothing": {
					"positive": ["dominatrix attire", "leather clothing"],
					"negative": [],
					"excludedBy": [hasCurse("ClothingRestrictionC")],
				}
			}
		},
		"Curse2020": {
			"affects": {
				"clothing": {
					"positive": ["(large glasses)", "(glasses:1.2)"],
					"negative": [],
					"excludedBy": [],
				}
			}
		},
		// "ComicRelief": {
		// 	"affects": {
		// 		"behaviour": {
		// 		"positive": [],
		// 		"negative": [],
		// 		"excludedBy": [],
		// 		}
		// 	}
		// },
		// "EqualOpportunity": {
		// 	"affects": {
		// 		"behaviour": {
		// 		"positive": [],
		// 		"negative": [],
		// 		"excludedBy": [],
		// 		}
		// 	}
		// },
		// "AbsolutePregnancy": {
		// 	"affects": {
		// 		"pregnancy": {
		// 		"positive": ["(Impregnation:1.5)"],
		// 		"negative": [],
		// 		"excludedBy": [],
		// 		}
		// 	}
		// },
		// "AbsoluteBirthControl": {
		// 	"affects": {
		// 		"pregnancy": {
		// 		"positive": [],
		// 		"negative": [],
		// 		"excludedBy": [],
		// 		}
		// 	}
		// },
		// "WackyWombs": {
		// 	"affects": {
		// 		"pregnancy": {
		// 		"positive": [],
		// 		"negative": [],
		// 		"excludedBy": [],
		// 		}
		// 	}
		// },
		"Omnitool": {
			"affects": {
				"pregnancy": {
					"positive": [],
					"negative": [],
					"excludedBy": [],
				}
			}
		},
		"Gooey": {
			"affects": {
				"gender": {
					"positive": ["(slime girl:1.4)"],
					"negative": [],
					"excludedBy": []
				},
				"skin": {
					"positive": ["(slime girl skin:1.3)", "gooey skin"],
					"negative": [],
					"excludedBy": []
				},
				"hair": {
					"positive": ["slime girl hair", "gooey hair"],
					"negative": [],
					"excludedBy": [],
				},
				"eyes": {
					"positive": ["gooey eyes"],
					"negative": [],
					"excludedBy": [],
				},
				"ears": {
					"positive": ["gooey ears"],
					"negative": [],
					"excludedBy": [],
				},
				"mouth":{
					"positive": ["(slime girl tongue:1.1)"],
					"negative": [],
					"excludedBy": [bool.closedMouth]
				},
				"breasts": {
					"positive": ["slime girl breasts", "gooey breasts"],
					"negative": [],
					"excludedBy": [],
				},
				"torso": {
					"positive": ["slime girl torso", "slime torso", "gooey torso"],
					"negative": [],
					"excludedBy": [],
				},
				"penis": {
					"positive": ["slimly penis", "gooey penis"],
					"negative": [],
					"excludedBy": [!range.penis],
				},
				// "vulva": {
				// 	"positive": ["slime girl pussy", "gooey pussy"],
				// 	"negative": [],
				// 	"excludedBy": [],
				// }
			}
		},
		"DoublePepperoni": {
			"affects": {
				"nipples": {
					"positive": ["(puffy_nipples:1.4)","(large areolae:1.4)","(long nipples:1.4)"],
					"negative": ["normal nipples"],
					"excludedBy": [bool.modestClothing, bool.normalClothing],
				}
			}
		},
		"LiteralBlushingVirgin": {
			"affects": {
				"behaviour": {
					"positive": ["timid", "blushing"],
					"negative": [],
					"excludedBy": [],
				}
			}
		},
		"HypnoHappytime": {
			"affects": {
				"behaviour": {
					"positive": ["(hypnotised eyes:1.3)", "(relaxed:1.3)"],
					"negative": ["(pendulum:2)", "(coin:2)"],
					"excludedBy": [],
				},
				"mouth": {
					"positive": ["drooling"],
					"negative": [],
					"excludedBy": [],
				}
			}
		},
		// "CrossdressYourHeart": {
		// 	"affects": {
		// 		"clothing": {
		// 			"positive": [],
		// 			"negative": [],
		// 			"excludedBy": [nude]
		// 		}
		// 	}
		// },
		// "LieDetector": {
		// 	"affects": {
		// 		"behaviour": {
		// 			"positive": [],
		// 			"negative": [],
		// 			"excludedBy": [],
		// 		}
		// 	}
		// },
		"Megadontia": {
			"affects": {
				"mouth": {
					"positive": ["(fangs:1.2)", "(sharp teeth:1.2)"],
					"negative": [],
					"excludedBy": [bool.closedMouth],
				}
			}
		},
		"Softie": {
			"affects": {
				"penis": {
					"positive": [],
					"negative": [],
					"excludedBy": [],
				},
				"nipples": {
					"positive": ["inverted nipples"],
					"negative": [],
					"excludedBy": [],
				}
			}
		},
		"HardMode": {
			"affects": {
				"penis": {
					"positive": ["veiny penis", "(erection:1.2)"],
					"negative": [],
					"excludedBy": [!range.penis],
				},
				"nipples": {
					"positive": ["(long nipples:1.5)", "hard nipples:1.5"],
					"negative": ["normal nipples"],
					"excludedBy": [],
				}
			}
		},
		"LingualLeviathan": {
			"affects": {
				"mouth": {
				"positive": ["(long tongue:1.3)", "(flexible tongue)", "frog tongue"],
				"negative": ["(multiple tongues:1.3", "detached tongue:1.3"],
				"excludedBy": [bool.closedMouth],
				}
			}
		},
		"TippingtheScales": {
		"affects": {
			"skin": { 
				"positive": ["reptile scale", "fish scales", "dragon scales"],
				"negative": [],
				"excludedBy": [],
			},
			"hair": {
				"positive": ["rough hair", "shiny hair"],
				"negative": [],
				"excludedBy": [],
			}
		}
		},
		"Reptail": {
			"affects": {
				"tail": {
					"positive": ["scaled tail", "lizard tail"],
					"negative": ["two tails", "multiple tails"],
					"excludedBy": [],
				},
			},
		},
		"ColdBlooded": {
			"affects": {
				"behaviour": {
					"positive": ["shaking", "shivering"],
					"negative": [],
					"excludedBy": [],
				}
			}
		},
		"ClothingRestrictionC": {
			"affects": {
				"clothing": {
					"positive": [],
					"negative": ["pants", "shirt", "dress", "skirt", "shoes", "sandals", "blouse", "suit"],
					"excludedBy": [],
				}
			}
		},
		"MassacreManicure": {
			"affects": {
				"limbs": {
					"positive": ["sharp fingernails", "dangerous fingernails", "long fingernails", "(claws:1.3)"],
					"negative": [],
					"excludedBy": [],
				}
			}
		},
		"DoM": {
			"affects": {
				"skin": {
					"positive": ["(Masochist)","(bruised", "(beaten:1.3)", "(bleeding:1.3)", `(bleeding ${characterData.state.blood} blood:1.3)`],
					"negative": [],
					"excludedBy": [],
				}
			}
		},
		"FlowerPower": {
		"affects": {
			"skin": {
				"positive": ["(vines sprouting)", "flowers", "(exotic flower)", "hibiscus", "(dryad:1.3)", "climbing foliage"],
				"negative": [],
				"excludedBy": [],
			},
			"limbs": {
				"positive": ["vines", "flowers", "fantasy flower", "(leaves wrapping wrists:1.3)"],
				"negative": [],
				"excludedBy": [],
			},
			"hair": {
				"positive": ["(flowers in hair)","fantasy flower", "sprouts in hair", "(vines in hair:1.3)", "flower crown"],
				"negative": [],
				"excludedBy": [],
			}
		}
		},
		"Cellulose": {
			"affects": {
				"gender": {
					"positive": ["(dryad:1.3)","(plant girl:1.3)"],
					"negative": [],
					"excludedBy": [],
					}
			}
		},
		"Carapacian": {
			"affects": {
				"skin": {
				"positive": ["shell skin", "insect skin"],
				"negative": [],
				"excludedBy": [],
				},
				"limbs": {
				"positive": ["armour arms", "keratin arms", "chitin arms", "bone arms"],
				"negative": [],
				"excludedBy": [],
				},
			}
		},
		"WrigglyAntennae": {
			"affects": {
				"hair": {
					"positive": ["(two antennae:1.3)", "two antennae coming from hair", "two antennae on head", "2 antennae"],
					"negative": [],
					"excludedBy": [],
				}
			}
		},
		"SubmissivenessRectificationB": {
			"affects": {
				"clothing": {
					"positive": ["submissive apparel"],
					"negative": [],
					"excludedBy": [bool.nude],
				},
			}
		},
		// 		"behaviour": {
		// 			"positive": [blushing, timid look],
		// 			"negative": [],
		// 			"excludedBy": [],
		// 		}
		// 	}
		// },
		// "LactationRejuvenationB": {
		// 	"affects": {
		// 		"pregnancy": {
		// 			"positive": [],
		// 			"negative": [],
		// 			"excludedBy": [],
		// 		},
		// 		"breasts": {
		// 			"positive": [],
		// 			"negative": [],
		// 			"excludedBy": [],
		// 		}
		// 	}
		"Horny": {
			"affects": {
				"hair": {
					"positive": ["smooth horns in hair", "curved horns", "pointed horns"],
					"negative": [],
					"excludedBy": [],
				}
			}
		},
		"DrawingSpades": { 
			"affects": {
				"behaviour": {
					"positive": ["succubus tail", "(((demon tail)))", "long tail", "pointed tail", "curved tail", "spade tail"],
					"negative": ["multiple tails"],
					"excludedBy": [],
				}
			}
		},
		"Leaky": {
			"affects": {
				"penis": {
					"positive": ["viscous liquid", "sticky liquid", "falling liquid", "Overflow", "Leakage", "dripping penis"],
					"negative": [],
					"excludedBy": [bool.modestClothing, bool.normalClothing],
				},
				"vulva": {
					"positive": ["viscous liquid", "sticky liquid", "falling liquid", "", "dripping"],
					"negative": [],
					"excludedBy": [bool.modestClothing, bool.normalClothing],
				}
			}
		},
		// "WanderingHands": {
		// 	"affects": {
		// 		"behaviour": {
		//	 		"positive": [],
		// 			"negative": [],
		// 			"excludedBy": [],
		// 		}
		// 	}
		// },
		"SemenDemon": {
			"affects": {
				"mouth": {
					"positive": ["(smug:0.7)"],
					"negative": [],
					"excludedBy": [],
				},
				"mouth": {
					"positive": ["(cum in mouth:1.2)","facial"],
					"negative": [],
					"excludedBy": [],
				}
			}
		},
		// "Quota": {
		// 	"affects": {
		// 		"behaviour": {
		// 			"positive": [],
		// 			"negative": [],
		// 			"excludedBy": [],
		// 		}
		// 	},
		// },
		// "SharedSpace": {
		// 	"affects": {
		// 		"behaviour": {
		// 			"positive": [],
		// 			"negative": [],
		// 			"excludedBy": [],
		// 		}
		// 	}
		// },
		// "RandomOrgasms": {
		// 	"affects": {
		// 		"behaviour": {
		// 		"positive": ["spasm", "leg spasm", "squirting"],
		// 		"negative": [],
		// 		"excludedBy": [],
		// 		}
		// 	}
		// },
		"CreatureoftheNight": {
			"affects": {
				"skin": {
					"positive": ["pale skin"],
					"negative": [],
					"excludedBy": [],
				},
				"mouth": {
					"positive": ["(vampire fangs:1.2)"],
					"negative": [],
					"excludedBy": [],
				}
			}
		},
		"Minish-ish": {
			"affects": {
				"size": {
					"positive": ["pencil size", "minimoys size", "schtroumpf size", "cursed"],
					"negative": [],
					"excludedBy": [],
				}
			}
		},
		"Colossal-able": {
			"affects": {
				"size": {
					"positive": ["giant", "tower size", "building size"],
					"negative": [],
					"excludedBy": [],
				}
			}
		},
		// "UrineReamplificationB": {
		// 	"affects": {
		// 		"clothing": {
		// 			"positive": [],
		// 			"negative": [],
		// 			"excludedBy": [marusolution],
		// 		}
		// 	}
		// },
		"EyeonthePrize": {
			"affects": {
				"eyes": {
					"positive": ["white iris on one eye", "transparent iris on one eye", "one non-functional eye", "grey pupil on one eye"],
					"negative": ["two eyes"],
					"excludedBy": [characterData.booleans.moonwatcher],
				}
			}
		},
		// "DeafeningSilence": {
		// 	"affects": {
		// 		"ears": {
		// 			"positive": [broken, twisted, removed, remnant, hollow],
		// 			"negative": [],
		// 			"excludedBy": [],
		// 		}
		// 	}
		// },
		// "TaciturnTurnaround": {
		// 	"affects": {
		// 		"mouth": {
		// 			"positive": ["mutilated tongue", "ripped off tongue", "scarred tongue", "removed tongue"],
		// 			"negative": [],
		// 			"excludedBy": [],
		// 		}
		// 	}
		// },
		// "Ampu-Q-tie": {
		// 	"affects": {
		// 		"limbs": {
		// 			"positive": ["ripped off", "removed", "bits and pieces", "unprofessional", "painful", "non-functional"],
		// 			"negative": [],
		// 			"excludedBy": [],
		// 		}
		// 	}
		// },
		// "NoseGoes": {
		// 	"affects": {
		// 	}
		// },
		// "ArmArmy": {
		// 	"affects": {
		// 		"limbs": {
		// 			"positive": ["parallel", "thorax", "abdomen", "anatomy"],
		// 			"negative": [],
		// 			"excludedBy": [],
		// 		}
		// 	}
		// },
		// "ALittleExtra": {
		// 	"affects": {
		// 		"penis": {
		// 		"positive": [],
		// 		"negative": [],
		// 		"excludedBy": [],
		// 		},
		// 		"vulva": {
		// 		"positive": [],
		// 		"negative": [],
		// 		"excludedBy": [],
		// 		}
		// 	}
		// },
		"Null": {
			"affects" : {
				// "penis": {
				// 	"positive": [],
				// 	"negative": [genitalia],
				// 	"excludedBy": [],
				// },
				// "vulva": {
				// 	"positive": [],
				// 	"negative": [genitalia],
				// 	"excludedBy": [],
				// },
					"breasts": {
						"positive": ["smooth breasts", "(nippleless breasts:1.2)", "(blank breasts)", "(no nipples)", "(athelia)", "(removed nipples)"],
						"negative": ["(nipples:1.3)"],
						"excludedBy": [],
					}
			}
		},
		// "Seafolk": {
		// 	"affects": {
		// 		"limbs": {
		// 			"positive": ["mermaid", "fins", "gills"],
		// 			"negative": ["human legs"],
		// 			"excludedBy": [],
		// 		}
		// 	}
		// },
		// "TakenforGranite": {
		// 	"affects": {
		// 		"skin": {
		// 			"positive": ["stone skin", "granite", "igneous", "Basalt"],
		// 			"negative": [],
		// 			"excludedBy": [],
		// 		}
		// 	}
		// },
		// "DoubleTrouble": {
		// 	"affects": {}
		// },
	/*  "Conjoined": { combined, hole, unity , completed, biomass
		"affects": {}
		},*/
		"AdversePossession": {
			"affects": {
				"behaviour": {
					"positive": ["evil grin"],
					"negative": [],
					"excludedBy": [],
				},
				"eyes": {
					"positive": ["(glowing eyes:1.3)", "evil eyes", "demonic eyes"],
					"negative": [],
					"excludedBy": [],
				}
			}
		},
		"TicklyTentacles": {
			"affects": {
				"limbs": {
					"positive": ["tentacles growing out back", "soft tentacles", "octopus tentacles", "alien tentacles", "monstrous tentacles", "long tentacles", "prehensile tentacles"],
					"negative": [],
					"excludedBy": [],
				},
			}
		},
		// "Eye-scream": {
		// 	"affects": {
		// 		"body": {
		// 			"positive": ["extra eye on eye location", "alien extra eyes", "orb extra eyes"],
		// 			"negative": [],
		// 			"excludedBy": [],
		// 		}
		// 	}
		// },
		// "AMouthful": {
		// 	"affects": {
		// 		"mouth": {
		// 		"positive": ["monstrous second mouth", "fleshy second mouth", "sharp second mouth", "horror second mouth"],
		// 		"negative": [],
		// 		"excludedBy": [],
		// 		}
		// 	}
		// },
		"BelowTheVeil": {
			"affects": {
				"background": {
					"positive": ["fog", "mist", "enlightened", "lovecraftian"],
					"negative": [],
					"excludedBy": [],
				}
			}
		},
		"PrincessProtocol": {
			"affects": {
				"behaviour": {
					"positive": ["scared", "nervous", "timid"],
					"negative": [],
					"excludedBy": [],
				},
				"clothing": {
					"positive": ["Princess Attire", "dress", "crown", "jewelery", "frilly clothing"],
					"negative": [],
					"excludedBy": [hasCurse("ClothingRestrictionC"), hasCurse("ClothingRestrictionA")],
				}
			}
		},
		// "GestationJumpstart": {
		// 	"affects": {
		// 		"pregnancy": {
		// 			"positive": [],
		// 			"negative": [],
		// 			"excludedBy": [],
		// 		}
		// 	}
		// },
		"BimboBabble": {
			"affects": {
				"clothing": {
					"positive": ["stylish clothing"],
					"negative": [],
					"excludedBy": [characterData.booleans.nude],
				},
				"behaviour": {
					"positive": ["ditzy"],
					"negative": [],
					"excludedBy": [],
				}
			}
		}
	}
	characterData.cursePromptArray = [];
	characterData.excluded = [];
	Object.entries(characterData.cursePrompts).filter(([curseName]) => hasCurse(curseName))
		.forEach(([curseName, curseData]) => {
			Object.entries(curseData.affects || {})
			.forEach(([category, categoryData]) => {
				characterData.cursePromptArray[category] = characterData.cursePromptArray[category] || { positive: [], negative: [] };
				characterData.excluded[category] = characterData.excluded[category] || { positive: [], negative: [] };
				const excludedReasons = (categoryData.excludedBy || [])
				.map(condition => {return condition ? condition : null;}).filter(Boolean);
				['positive', 'negative'].forEach(type => {
					(categoryData[type] || []).forEach(tag => {
						excludedReasons.length === 0
						? characterData.cursePromptArray[category][type].push(tag)
						: characterData.excluded[category][type].push({
							curse: curseName,
							tag: tag,
						});
					});
				});
			});
		});
	// console.log(positive+",\n henlo \n"+negative);
	// console.log(characterData.cursePromptArray, "Preserved");
	// console.log(characterData.excluded, "Excluded");
	return characterData;
};

setup.compileCombinedPrompts = function(characterData) {
	const hasCurse = (curseName) => {
		return characterData.curses.some(curse => curse.constructor.name === curseName);
	};
	const bool = characterData.booleans;
	const range = characterData.range;
	const CATEGORY_ORDER = [
		'quality',
		'framing',
		'style',
		'misprints',
		'gender',
		'age',
		'skin',
		'face',
		'eyes',
		'nose',
		'ears',
		'mouth',
		'hair',
		'torso',
		'fitness',
		'limbs',
		'pregnancy',
		'breasts',
		'nipples',
		'penis',
		'vulva',
		'tail',
		'clothing',
		'behaviour',
		'libido',
		'background',
		'special',
	];

	const INTERMEDIATE_CONTENT = {
		'framing': ', <lora:Dalle3_AnimeStyle_PONY_Lora:1>,',
		'skin': 'BREAK',
		'limbs': 'BREAK',
		'vulva': 'BREAK',
		'libido': 'BREAK',
	};

	const SKIP_CATEGORIES = {
		"nipples": [
			() => bool.modestClothing,
			() => bool.normalClothing,
			() => bool.immodestClothing,
			() => hasCurse("Null")
		],
		"penis": [
			() => bool.modestClothing,
			() => bool.normalClothing,
			() => bool.immodestClothing,
			() => bool.skimpyClothing,
			() => !range.penis
		],
		"vulva": [
			() => bool.modestClothing,
			() => bool.normalClothing,
			() => bool.immodestClothing,
			() => bool.skimpyClothing,
			() => hasCurse("Null")
		],
	};

	const shouldSkipCategory = (category) => {
		if (!SKIP_CATEGORIES[category]) return false;
		return SKIP_CATEGORIES[category].some(condition => condition());
	};

	const categoryPositivePrompts = {};
	const categoryNegativePrompts = {};
	const cursePromptArray = characterData.cursePromptArray || {};
	const statPromptArray = characterData.statPromptArray || {};

	const removeConflictingPrompts = (sourceArray, targetArray) => {
		const cleanedTargetArray = { ...targetArray };
		Object.entries(sourceArray).forEach(([sourceCategory, sourcePrompts]) => {
			const oppositeType = sourcePrompts.positive ? 'negative' : 'positive';
			if (cleanedTargetArray[sourceCategory]) {
				const sourcePromptsToCheck = sourcePrompts.positive || sourcePrompts.negative;
				const targetPromptsToCheck = cleanedTargetArray[sourceCategory][oppositeType];
				if (sourcePromptsToCheck && targetPromptsToCheck) {
					cleanedTargetArray[sourceCategory][oppositeType] = targetPromptsToCheck.filter(
						targetPrompt => !sourcePromptsToCheck.includes(targetPrompt)
					);
				}
			}
		});
		return cleanedTargetArray;
	};

	const cleanedStatPromptArray = removeConflictingPrompts(cursePromptArray, statPromptArray);

	const processPromptArray = (sourceArray, targetPositive, targetNegative) => {
		Object.entries(sourceArray).forEach(([category, prompts]) => {
			if (CATEGORY_ORDER.includes(category) && !shouldSkipCategory(category)) {
				if (!targetPositive[category]) targetPositive[category] = [];
				if (!targetNegative[category]) targetNegative[category] = [];

				if (prompts.positive && Array.isArray(prompts.positive)) {
					targetPositive[category].push(...prompts.positive);
				}
				if (prompts.negative && Array.isArray(prompts.negative)) {
					targetNegative[category].push(...prompts.negative);
				}
			}
		});
	};

	processPromptArray(cursePromptArray, categoryPositivePrompts, categoryNegativePrompts);
	processPromptArray(cleanedStatPromptArray, categoryPositivePrompts, categoryNegativePrompts);

	const combinedPositivePrompts = [];
	const combinedNegativePrompts = [];

	CATEGORY_ORDER.forEach(category => {
		if (shouldSkipCategory(category)) {
			// console.log(`Skipping category: ${category}`);
			return;
		}

		const positiveCategoryPrompts = [];
		const negativeCategoryPrompts = [];

		if (categoryPositivePrompts[category]) {
			const uniquePositive = [...new Set(categoryPositivePrompts[category])];
			positiveCategoryPrompts.push(uniquePositive.join(", "));
		}

		if (INTERMEDIATE_CONTENT[category] && positiveCategoryPrompts.length > 0) {
			positiveCategoryPrompts.push(INTERMEDIATE_CONTENT[category]);
		} else if (positiveCategoryPrompts.length > 0) {
			positiveCategoryPrompts[positiveCategoryPrompts.length - 1] += ",";
		}

		if (positiveCategoryPrompts.length > 0) {
			combinedPositivePrompts.push(positiveCategoryPrompts.join(" "));
		}

		if (categoryNegativePrompts[category]) {
			const uniqueNegative = [...new Set(categoryNegativePrompts[category])];
			negativeCategoryPrompts.push(uniqueNegative.join(", "));
		}

		if (negativeCategoryPrompts.length > 0) {
			negativeCategoryPrompts[negativeCategoryPrompts.length - 1] += ",";
		}

		if (negativeCategoryPrompts.length > 0) {
			combinedNegativePrompts.push(negativeCategoryPrompts.join(" "));
		}
	});

	// Clean up repeated commas
	const cleanCommas = (str) => {
		return str
			.replace(/,\s*,+/g, ", ")
			.replace(/\s*,\s*$/, "");
	};

	const positive = cleanCommas(combinedPositivePrompts.join(" "));
	const negative = cleanCommas(combinedNegativePrompts.join(" "));

	// console.log("Compiled Prompts:");
	// console.log("Positive:", positive);
	// console.log("Negative:", negative);

	return [positive, negative];
};

setup.comfyUI_GenerateStandardParameters = function() {
	let characterData = setup.comfyUI_PrepareCharacterData();
	characterData = setup.comfyUI_GenerateStatParameters(characterData);
	characterData = setup.comfyUI_GenerateCurseParameters(characterData);
	let [positive, negative] = setup.compileCombinedPrompts(characterData);
	let checkpoint = "hassakuXLPony_v13BetterEyesVersion.safetensors";
	let steps = 30;
	let cfg = 10.0;
	let width = 1024;
	let height = 1024;
	var seed = null;
	if (setup.customSeedInput == null || setup.customSeedInput == "" || !Number.isInteger(parseInt(setup.customSeedInput))) {
		seed = SugarCube.State.prng.seed | Math.round(Math.random() * 10_000);
	} else {
		seed = parseInt(setup.customSeedInput);
	}
	return [positive, negative, checkpoint, steps, cfg, seed, width, height];
}

setup.comfyUI_GeneratePortraitWorkflow = async function() {

	if (SugarCube.State.variables.UseAdvancedComfyUIPrompt == true) {
		var [positive, negative, checkpoint, steps, cfg, seed, width, height] = setup.comfyUI_GenerateAdvancedParameters();
	} else {
		var [positive, negative, checkpoint, steps, cfg, seed, width, height] = setup.comfyUI_GenerateStandardParameters();
	}


	// clone workflow so it can be edited
	var workflow = null;
	if (SugarCube.State.variables.DisableTransparentPortraitBackground == true) {
		workflow = JSON.parse(JSON.stringify(SIMPLE_T2I_PORTRAIT_WORKFLOW));
	} else {
		workflow = JSON.parse(JSON.stringify(SIMPLE_T2I_PORTRAIT_WORKFLOW_REMOVE_BACKGROUND));
	}

	if (setup.customPromptPrefix != null) {
		positive = setup.customPromptPrefix + "," + positive
	}

	if (setup.customPromptSuffix != null) {
		positive = positive + "," + setup.customPromptSuffix
	}

	workflow["5"]["inputs"]["ckpt_name"] = checkpoint
	workflow["9"]["inputs"]["text"] = positive
	workflow["10"]["inputs"]["text"] = negative
	workflow["11"]["inputs"]["steps"] = steps
	workflow["11"]["inputs"]["cfg"] = cfg
	workflow["11"]["inputs"]["seed"] = seed
	workflow["12"]["inputs"]["width"] = width
	workflow["12"]["inputs"]["height"] = height

	return workflow;
}

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

var is_generation_busy = false;
var last_workflow = null;
setup.comfyUI_GeneratePortrait = async function() {
	const notificationElement = document.getElementById('notification');

	if (is_generation_busy) {
		console.log("generation busy");
		notificationElement.style.display = "block";
		notificationElement.textContent = "An image is already being generated.";
		return;
	}
	is_generation_busy = true;

	notificationElement.style.display = "none";

	// log outputted workflow
	// console.log(workflow);

	// request to the proxy to generate the portrait
	let data = null;
	try {
		const workflow = await setup.comfyUI_GeneratePortraitWorkflow();
		// if (last_workflow == JSON.stringify(workflow)) {
		// 	is_generation_busy = false;
		// 	return; // already the same
		// }
		// workflow was updated
		last_workflow = JSON.stringify(workflow);
		data = await generateWorkflowImage(workflow);
	} catch (error) {
		console.error('Unable to invoke ComfyUI generator: ', error);
		is_generation_busy = false;
		notificationElement.style.display = "block";
		notificationElement.textContent = "Unable to contact the ComfyUI proxy. Make sure the Python code is running! Check the one-click installer terminal. " + error;
		return;
	}

	// console.log(data);

	// check if we actually received any images
	if (data.images == null || data.images.length == 0) {
		console.error('No images returned from server. This might be due to an issue with the Stable Diffusion model or the server.');
		is_generation_busy = false;
		notificationElement.style.display = "block";
		notificationElement.textContent = "No images were returned from the proxy! Is ComfyUI running? Check the one-click installer terminal.";
		return;
	}

	// once we receive the image, save it as the player portrait
	is_generation_busy = false;

	var storeKey = "dalleImage";
	var b64Image = data.images[0];
	// console.log("Base64 Data Length: ", b64Image.length);
	try {
		setup.storeImage(storeKey, b64Image);
		// console.log('Image successfully stored.');
	} catch(error) {
		console.error('Failed to store image due to error:', error);
		notificationElement.style.display = "block";
		throw new Error('Failed to store image due to error: ' + error);
	}
}
