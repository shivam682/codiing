import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from typing import List, Dict

# Initialize embedding model
embed_model = SentenceTransformer("BAAI/bge-large-en")
embedding_dim = embed_model.get_sentence_embedding_dimension()

# File paths
FAISS_INDEX_PATH = "cr_index.faiss"
META_PATH = "cr_metadata.pkl"


def embed_texts(texts: List[str]):
    return embed_model.encode(texts, normalize_embeddings=True)


def build_faiss_index(embeddings):
    index = faiss.IndexFlatIP(embedding_dim)  # Inner product = cosine similarity if normalized
    index.add(embeddings)
    return index


def save_index(index, metadata):
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump(metadata, f)


def load_index():
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(META_PATH, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata


def index_crs(cr_entries: List[Dict]):
    """
    cr_entries: List of dicts with keys: 'cr_number', 'summary', 'signature', 'callstack'
    """
    combined_texts = []
    metadata = []

    for entry in cr_entries:
        full_text = f"{entry['signature']}\n{entry.get('callstack', '')}\n{entry.get('summary', '')}"
        combined_texts.append(full_text)
        metadata.append({
            "cr_number": entry["cr_number"],
            "summary": entry.get("summary", ""),
            "signature": entry["signature"]
        })

    embeddings = embed_texts(combined_texts)
    index = build_faiss_index(embeddings)
    save_index(index, metadata)
    print(f"✅ Indexed {len(cr_entries)} CRs and saved to disk.")


def search_crs(query_text: str, top_k=5):
    index, metadata = load_index()
    query_embedding = embed_texts([query_text])[0].reshape(1, -1)
    scores, indices = index.search(query_embedding, top_k)

    results = []
    for idx, score in zip(indices[0], scores[0]):
        if idx < len(metadata):
            results.append({
                "cr_number": metadata[idx]["cr_number"],
                "summary": metadata[idx]["summary"],
                "signature": metadata[idx]["signature"],
                "score": round(float(score), 4)
            })

    return results


# --------- Example Usage ---------
if __name__ == "__main__":
    import json

    # Example: indexing
    cr_data = [
        {
            "cr_number": "CR123",
            "signature": "Crash in module X due to null ptr",
            "callstack": "main -> init -> crash_here()",
            "summary": "Null pointer dereference in module X"
        },
        {
            "cr_number": "CR456",
            "signature": "Segfault in Y when accessing buffer",
            "callstack": "main -> handler -> buffer_overflow()",
            "summary": "Buffer overrun in handler"
        }
    ]
    index_crs(cr_data)

    # Example: searching
    query = "Segmentation fault in buffer handler"
    top_matches = search_crs(query, top_k=3)
    print(json.dumps(top_matches, indent=2))
