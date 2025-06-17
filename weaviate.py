# cr_dedup_utils.py
import os
import weaviate
from typing import List, Tuple
from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Initialize models
embedding_model = OpenAIEmbeddings()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# Initialize Weaviate client (assuming local Weaviate running)
WEAVIATE_URL = "http://localhost:8080"
client = weaviate.Client(WEAVIATE_URL)


# Ensure schema exists
def ensure_weaviate_schema():
    schema = {
        "classes": [
            {
                "class": "CrashLog",
                "properties": [
                    {"name": "jira_id", "dataType": ["text"]},
                    {"name": "log_type", "dataType": ["text"]},
                    {"name": "filename", "dataType": ["text"]},
                    {"name": "chunk_index", "dataType": ["int"]},
                    {"name": "content", "dataType": ["text"]}
                ]
            }
        ]
    }
    if not client.schema.contains(schema):
        client.schema.create(schema)


def summarize_jira(signature: str, callstack: str) -> List[float]:
    """Create embedding from signature + callstack summary"""
    summary = f"Signature: {signature}\nCallstack: {callstack}"
    return embedding_model.embed_query(summary)


def find_top_k_similar(summary_embedding: List[float], k: int = 5) -> List[Tuple[str, float]]:
    """Return top-k similar past Jira IDs from Weaviate"""
    response = client.query.get("CrashLog", ["jira_id"])
    response = response.with_near_vector({"vector": summary_embedding}).with_limit(k).do()
    results = []
    for item in response['data']['Get']['CrashLog']:
        results.append((item['jira_id'], 1.0))  # score is placeholder
    return results


def extract_logs_used_in_cr(cr_json_path: str) -> List[str]:
    """Parse CR metadata to extract relevant log types (e.g. ['tx_ring', 'registers'])"""
    import json
    with open(cr_json_path, 'r') as file:
        cr_info = json.load(file)
    return cr_info.get("logs_checked", [])


def chunk_and_embed_selected_logs(jira_id: str, log_dir: str, log_types: List[str]):
    """Chunk and upload selected logs from new crash to Weaviate"""
    ensure_weaviate_schema()
    for fname in os.listdir(log_dir):
        for log_type in log_types:
            if log_type in fname:
                with open(os.path.join(log_dir, fname), 'r', errors='ignore') as f:
                    content = f.read()
                chunks = text_splitter.split_text(content)
                for i, chunk in enumerate(chunks):
                    embedding = embedding_model.embed_query(chunk)
                    client.data_object.create(
                        data_object={
                            "jira_id": jira_id,
                            "log_type": log_type,
                            "filename": fname,
                            "chunk_index": i,
                            "content": chunk
                        },
                        class_name="CrashLog",
                        vector=embedding
                    )


def clear_temp_crash_logs(jira_id: str):
    """Delete old logs from Weaviate for a specific Jira ID (used for temp crash storage)"""
    client.batch.delete_objects(
        class_name="CrashLog",
        where={
            "path": ["jira_id"],
            "operator": "Equal",
            "valueText": jira_id
        }
    )
