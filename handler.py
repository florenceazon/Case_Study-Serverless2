# handler.py
from diffusers import DiffusionPipeline
import torch
import base64
from io import BytesIO
from PIL import Image

# Load model
try:
pipe = DiffusionPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")
except Exception as e:
    print("Error loading pipeline:", e)
    pipe = None

print("CUDA available:", torch.cuda.is_available())

def handler(event):
    if pipe is None:
        return {"error": "Pipeline failed to load"}
    prompt = event.get("input", "A surreal landscape with floating islands")

    try:
    # Generate image
    image = pipe(prompt).images[0]

    # Encode image to base64
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return {"output": img_str}
    except Exception as e:
        return {"error": str(e)}
