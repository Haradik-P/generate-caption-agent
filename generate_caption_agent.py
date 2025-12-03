from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import torch
import sys

# -----------------------------
# Setup device
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# -----------------------------
# Load BLIP model (once)
# -----------------------------
print("Loading BLIP model...")
processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-flan-t5-xl",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)
print(f"Model loaded on {device}!")

# -----------------------------
# Function to generate social media caption
# -----------------------------
def generate_social_caption(image_path):
    """
    Input: path to image
    Output: detailed description + suggested hashtags
    """
    try:
        # Load image
        image = Image.open(image_path).convert("RGB")

        # Create prompt for detailed description
        prompt = "Describe this image in detail for a social media post, and suggest 5 relevant hashtags."

        # Prepare inputs
        inputs = processor(image, text=prompt, return_tensors="pt").to(device)

        # Generate output
        output_ids = model.generate(**inputs, max_length=100)
        caption = processor.decode(output_ids[0], skip_special_tokens=True)

        return caption
    except Exception as e:
        return f"Error: {str(e)}"

# -----------------------------
# Command-line usage
# -----------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_caption_agent.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    caption = generate_social_caption(image_path)
    print("\n--- Generated Caption ---\n")
    print(caption)
    print("\n-------------------------\n")
