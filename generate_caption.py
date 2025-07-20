import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# Load BLIP model
# device = "cuda" if torch.cuda.is_available() else "cpu"
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Caption generator
def generate_caption(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        output = model.generate(**inputs, max_length=50)
        caption = processor.decode(output[0], skip_special_tokens=True)
    return caption

# Example usage
if __name__ == "__main__":
    img_path = "images/0001.jpg"  # change to any image
    caption = generate_caption(img_path)
    print(f"🖼️ Caption: {caption}")
