# Parallel Chunking + Qdrant Insertion + Best Match Retrieval Pipeline

import os
import json
import uuid
import multiprocessing
from typing import List, Tuple

from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant as QdrantStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from rapidfuzz import fuzz
from difflib import SequenceMatcher

# CONFIG
DUMP_DIR = "/mnt/data/new_jira_dump"
COLLECTION_NAME = "log_chunks"
QDRANT_URL = "http://localhost:6333"
CHUNK_SIZE = 15
CHUNK_OVERLAP = 5
BATCH_SIZE = 1000
NUM_WORKERS = 4

embedding_model = OpenAIEmbeddings()
qdrant_client = QdrantClient(url=QDRANT_URL)

# Step 1: Chunking

def chunk_file(file_path: str, chunk_size: int, overlap: int) -> List[str]:
    with open(file_path, "r", encoding="utf-8", errors='ignore') as f:
        lines = f.readlines()
    chunks = []
    for i in range(0, len(lines), chunk_size - overlap):
        chunk = lines[i:i + chunk_size]
        if chunk:
            chunks.append("".join(chunk).strip())
    return chunks

def chunk_and_embed_worker(file_paths: List[str]) -> List[Tuple[str, dict]]:
    results = []
    for fpath in file_paths:
        try:
            chunks = chunk_file(fpath, CHUNK_SIZE, CHUNK_OVERLAP)
            embeddings = embedding_model.embed_documents(chunks)
            for chunk, vector in zip(chunks, embeddings):
                results.append((chunk, {"vector": vector, "metadata": {"file": os.path.basename(fpath), "id": str(uuid.uuid4())}}))
        except Exception as e:
            print(f"Error in file {fpath}: {e}")
    return results

# Step 2: Batch insert into Qdrant

def insert_chunks_to_qdrant(chunks_and_vectors):
    if not qdrant_client.collection_exists(COLLECTION_NAME):
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )

    from itertools import islice
    def batch(iterable, size):
        it = iter(iterable)
        while True:
            batch = list(islice(it, size))
            if not batch:
                break
            yield batch

    for b in batch(chunks_and_vectors, BATCH_SIZE):
        payload = [item[1]["metadata"] for item in b]
        vectors = [item[1]["vector"] for item in b]
        texts = [item[0] for item in b]
        QdrantStore(
            client=qdrant_client,
            collection_name=COLLECTION_NAME,
            embedding=embedding_model
        ).add_texts(texts, metadatas=payload)

# Step 3: Search + Best Match using Fuzzy/SeqMatcher

def retrieve_best_match(snippet: str) -> str:
    store = QdrantStore(
        client=qdrant_client,
        collection_name=COLLECTION_NAME,
        embedding=embedding_model,
    )
    results = store.similarity_search(snippet, k=10)
    best = (None, 0)
    for res in results:
        fuzz_score = fuzz.partial_ratio(snippet, res.page_content)
        seq_score = SequenceMatcher(None, snippet, res.page_content).ratio() * 100
        final_score = max(fuzz_score, seq_score)
        if final_score > best[1]:
            best = (res.page_content, final_score)
    return f"Best match [Score: {best[1]:.2f}]:\n{best[0]}"

if __name__ == "__main__":
    from multiprocessing import Pool
    all_files = [os.path.join(DUMP_DIR, f) for f in os.listdir(DUMP_DIR)]
    split = [all_files[i::NUM_WORKERS] for i in range(NUM_WORKERS)]

    print("Chunking and embedding in parallel...")
    with Pool(NUM_WORKERS) as p:
        results = p.map(chunk_and_embed_worker, split)

    flat_results = [item for sublist in results for item in sublist]
    print(f"Total chunks to insert: {len(flat_results)}")
    insert_chunks_to_qdrant(flat_results)
    print("Data inserted into Qdrant.")

    # Example usage
    example_snippet = "Example crash trace or log snippet here..."
    print(retrieve_best_match(example_snippet))


I've updated your code to include:

Parallel chunking of files from the dump directory

Embedding and storing them in Qdrant vector DB

Retrieving top 10 similar chunks for a given log snippet

Selecting the best match using fuzzy and sequence matcher scores


Let me know if you want a variation with FAISS instead of Qdrant, or want to persist metadata for better traceability.

