import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
import pickle
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# 1. Setup device and model
# device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 2. Image folder and batch config
image_folder = "images"
batch_size = 32

# 3. Load image paths
image_paths = sorted([
    os.path.join(image_folder, fname)
    for fname in os.listdir(image_folder)
    if fname.lower().endswith((".jpg", ".jpeg", ".png"))
])

all_embeddings = []
valid_paths = []

# 4. Batch process images
for i in tqdm(range(0, len(image_paths), batch_size)):
    batch_paths = image_paths[i:i + batch_size]
    images = []

    for path in batch_paths:
        try:
            img = Image.open(path).convert("RGB")
            images.append(img)
            valid_paths.append(path)
        except:
            print(f"Skipping corrupt image: {path}")
            continue

    if not images:
        continue

    inputs = processor(images=images, return_tensors="pt", padding=True)

    with torch.no_grad():
        embeddings = model.get_image_features(**inputs)
        embeddings = embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)  # Normalize
        all_embeddings.append(embeddings.cpu())

# 5. Stack and save embeddings
embeddings = torch.cat(all_embeddings).numpy()
np.save("image_embeddings.npy", embeddings)

# 6. Save image paths
with open("image_paths.pkl", "wb") as f:
    pickle.dump(valid_paths, f)

print(f"✅ Saved embeddings for {len(valid_paths)} images.")
