# handler.py
from diffusers import DiffusionPipeline
import torch
import base64
from io import BytesIO
from PIL import Image

# Your Hugging Face access token
access_token = "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"

# Load model
pipe = DiffusionPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.float16,
    use_auth_token=access_token
).to("cuda")

def handler(event, context):
    # Ask the user for their prompt (this will run in the terminal)
    user_input = input("Please enter a word for the image generation: ")

    # If no input is given, use a default prompt
    prompt = user_input if user_input else "Apple"
    #prompt = event.get("input", "A surreal landscape with floating islands")
    
    # Generate image
    image = pipe(prompt).images[0]

    # Encode image to base64
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return {"output": img_str}
