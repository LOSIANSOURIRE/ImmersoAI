import torch
from diffusers import StableDiffusionPipeline
from pathlib import Path
import json
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
    "/mnt/local/ImmersoAI/Dreamlike_Textual_Pipeline/textual_inversion_dreamlike_diffusion",
    token="<baji-face>",
)

data = []
with open('/mnt/local/ImmersoAI/dataset.json') as f:
    for line in f:
        data.append(json.loads(line))

subjects = []
bajirao_templates = []
for i in range(len(data)):
    subjects.append(data[i]['image'])
    bajirao_templates.append(data[i]['caption'].replace('sks rajinikanth', '<baji-face> Ranveer Singh')) #Change as per used

output_dir = Path("/mnt/local/ImmersoAI/Dreamlike_Textual_Pipeline/textual_inversion_dreamlike_diffusion_immerso_prompt")
output_dir.mkdir(parents=True, exist_ok=True)

for subject, prompt in zip(subjects, bajirao_templates):
    image = pipeline(
        prompt=prompt,
        num_inference_steps=50,
        guidance_scale=7.5
    ).images[0]

    image.save(output_dir / f"{subject}")
    print(f"Saved: {subject} | Prompt: {prompt}")