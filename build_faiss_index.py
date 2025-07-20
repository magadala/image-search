import numpy as np
import faiss

# 1. Load image embeddings
embeddings = np.load("image_embeddings.npy").astype("float32")
faiss.normalize_L2(embeddings)  # Required for cosine similarity

# 2. Build FAISS index
index = faiss.IndexFlatIP(embeddings.shape[1])  # IP = cosine after L2 normalization
index.add(embeddings)

# 3. Save index
faiss.write_index(index, "faiss_index.index")

print(f"✅ FAISS index created and saved with {index.ntotal} vectors.")
