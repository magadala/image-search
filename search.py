import torch
import faiss
import numpy as np
import pickle
from transformers import CLIPProcessor, CLIPModel
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# 1. Load model and index
# device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
index = faiss.read_index("faiss_index.index")

# 2. Load image paths
with open("image_paths.pkl", "rb") as f:
    image_paths = pickle.load(f)

# 3. Search function
def search_images(query, top_k=5):
    inputs = processor(text=[query], return_tensors="pt", padding=True)
    with torch.no_grad():
        query_embed = model.get_text_features(**inputs)
        query_embed = query_embed / query_embed.norm(p=2, dim=-1, keepdim=True)
        query_embed = query_embed.cpu().numpy().astype("float32")

    faiss.normalize_L2(query_embed)
    scores, indices = index.search(query_embed, top_k)

    results = [(image_paths[i], float(scores[0][j])) for j, i in enumerate(indices[0])]
    return results

# 4. Example usage
if __name__ == "__main__":
    query = "a man riding a horse"
    results = search_images(query)

    print(f"\n🔍 Top results for: '{query}'\n")
    for path, score in results:
        print(f"{score:.4f} → {path}")
