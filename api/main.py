import torch
import pickle
import faiss
import numpy as np
from PIL import Image
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity
import os

ENABLE_CAPTIONING = True  # Set to False to disable BLIP captions
app = FastAPI()

# Load models
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

if ENABLE_CAPTIONING:
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load FAISS index and image embeddings
faiss_index = faiss.read_index("faiss_index.index")
with open("image_paths.pkl", "rb") as f:
    image_paths = pickle.load(f)
image_embeddings = np.load("image_embeddings.npy")

class Query(BaseModel):
    text: str

def rewrite_query(query: str) -> str:
    if len(query.split()) < 3:
        return f"a photo of {query.strip()}"
    return query.strip()

def generate_caption(image_path):
    raw_image = Image.open(image_path).convert("RGB")
    inputs = blip_processor(raw_image, return_tensors="pt")
    out = blip_model.generate(**inputs)
    return blip_processor.decode(out[0], skip_special_tokens=True)

def rerank_results(query_text, results):
    inputs = clip_processor(text=[query_text], return_tensors="pt", padding=True)
    with torch.no_grad():
        query_embed = clip_model.get_text_features(**inputs).numpy()
    query_embed = query_embed / np.linalg.norm(query_embed, axis=1, keepdims=True)

    for result in results:
        caption = result["caption"]
        cap_input = clip_processor(text=[caption], return_tensors="pt", padding=True)
        with torch.no_grad():
            cap_embed = clip_model.get_text_features(**cap_input).numpy()
        cap_embed = cap_embed / np.linalg.norm(cap_embed, axis=1, keepdims=True)
        sim = cosine_similarity(query_embed, cap_embed)[0][0]
        result["caption_score"] = float(sim)

    return sorted(results, key=lambda x: x["caption_score"], reverse=True)[:5]

@app.post("/search")
def search(query: Query):
    rewritten_query = rewrite_query(query.text)

    # Embed the query
    inputs = clip_processor(text=[rewritten_query], return_tensors="pt", padding=True)
    with torch.no_grad():
        query_embedding = clip_model.get_text_features(**inputs).numpy()
    query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)

    # Search with FAISS
    _, indices = faiss_index.search(query_embedding, k=10)

    results = []
    for i, idx in enumerate(indices[0]):
        image_path = image_paths[idx]
        score = float(np.dot(query_embedding, image_embeddings[idx]))

        if ENABLE_CAPTIONING:
            caption = generate_caption(image_path)
            explanation = f"The image is relevant because it shows: '{caption}', which matches the query: '{rewritten_query}'."
        else:
            caption = "BLIP captioning disabled"
            explanation = f"Result {i+1} matches the query: '{rewritten_query}'"

        results.append({
            "image_path": image_path,
            "score": score,
            "caption": caption,
            "explanation": explanation
        })

    if ENABLE_CAPTIONING:
        results = rerank_results(rewritten_query, results)

    return results
