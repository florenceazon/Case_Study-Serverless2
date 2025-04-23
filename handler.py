# handler.py
from diffusers import DiffusionPipeline
import torch
import base64
from io import BytesIO
from PIL import Image

# Load model
pipe = DiffusionPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")

def handler(event):
    prompt = event.get("input", "A surreal landscape with floating islands")
    
    # Generate image
    image = pipe(prompt).images[0]

    # Encode image to base64
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return {"output": img_str}
