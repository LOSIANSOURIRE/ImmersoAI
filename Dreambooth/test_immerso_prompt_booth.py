import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from pathlib import Path
import json
# Load SDXL pipeline with safety checker disabled
model_directory = "/mnt/local/ImmersoAI/Dreamlike_Diffusion_Pipeline/training/dreambooth-concept1"
pipeline = DiffusionPipeline.from_pretrained(
        model_directory,
        scheduler = DPMSolverMultistepScheduler.from_pretrained(model_directory, subfolder="scheduler"),
        torch_dtype=torch.float16,
    ).to("cuda")



data = []
with open('/mnt/local/ImmersoAI/dataset.json') as f:
    for line in f:
        data.append(json.loads(line))

subjects = []
bajirao_templates = []
for i in range(len(data)):
    subjects.append(data[i]['image'])
    bajirao_templates.append(data[i]['caption'].replace('sks rajinikanth', '<baji-face>'))

output_dir = Path(f"{model_directory}_immerso_prompt_photo_output")
output_dir.mkdir(parents=True, exist_ok=True)

for subject, prompt in zip(subjects, bajirao_templates):
    image = pipeline(
        prompt=prompt,
        num_inference_steps=50,
        guidance_scale=7.5
    ).images[0]

    image.save(output_dir / f"{subject}")
    print(f"Saved: {subject} | Prompt: {prompt}")