import torch
from diffusers import StableDiffusionPipeline
from pathlib import Path

# Load SDXL pipeline with safety checker disabled
pipeline = StableDiffusionPipeline.from_pretrained(
    "dreamlike-art/dreamlike-photoreal-2.0",
    torch_dtype=torch.float16,
    # variant="fp16",
    safety_checker=None,  # You can leave this enabled if needed
    use_safetensors=True
).to("cuda")

# Load learned embeddings
pipeline.load_textual_inversion(
    "/mnt/local/ImmersoAI/Dreamlike_Textual_Pipeline/textual_inversion_copy_upscaled1.5_photoreal",
    token="<baji-face>",
)

# Prompt setup
bajirao_templates = [
    "a cinematic photo of <baji-face> from Bajirao Mastani",
    "a dramatic photo of <baji-face> wearing Maratha armor",
    "a movie still of <baji-face> in a battlefield scene",
    "a historical portrait of <baji-face> with royal attire",
    "a close-up of <baji-face> with a fierce expression",
    "a dynamic shot of <baji-face> as Bajirao",
    "a royal photo of <baji-face> with a sword",
]

subjects = ["bajirao1","bajirao2", "bajirao3", "bajirao4", "bajirao5", "bajirao6", "bajirao7"]

output_dir = Path("/mnt/local/ImmersoAI/Dreamlike_Textual_Pipeline/textual_inversion_copy_upscaled1.5_photoreal_photo")
output_dir.mkdir(parents=True, exist_ok=True)

for subject, prompt in zip(subjects, bajirao_templates):
    # SDXL needs both positive and negative prompt
    image = pipeline(
        prompt=prompt,
        num_inference_steps=50,
        guidance_scale=7.5
    ).images[0]

    image.save(output_dir / f"{subject}.png")
    print(f"Saved: {subject}.png | Prompt: {prompt}")