import faiss
import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz

EMBED_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
FAISS_INDEX_PATH = "cr_faiss.index"
META_PATH = "cr_metadata.json"

# Global in-memory index and metadata
index = None
metadata = []

def embed(texts):
    return EMBED_MODEL.encode(texts, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)

def index_crs(cr_list):
    global index, metadata

    texts = [cr["text"] for cr in cr_list]
    meta = [cr["metadata"] for cr in cr_list]
    embeddings = embed(texts)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    metadata = meta

    # Save
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f)

def load_index():
    global index, metadata
    if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(META_PATH):
        return
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

def search_crs(query, top_k=3, faiss_k=20):
    if index is None:
        load_index()
    query_vec = embed([query])
    D, I = index.search(query_vec, faiss_k)

    candidates = []
    for idx in I[0]:
        if idx < len(metadata):
            cr_text = metadata[idx].get("text", "")
            score = fuzz.partial_ratio(query, cr_text)
            candidates.append({
                "score": score,
                "cr_number": metadata[idx].get("cr_number"),
                "text": cr_text,
                "source": metadata[idx].get("source", "N/A")
            })

    # Return top N by fuzzy match
    return sorted(candidates, key=lambda x: x["score"], reverse=True)[:top_k]
