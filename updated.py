from langgraph.graph import END, StateGraph
from langchain.agents import Tool
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Weaviate

import os
import json
import weaviate
import openai

# --- CONFIG ---
openai.api_key = "YOUR_OPENAI_API_KEY"
CR_JSON_FOLDER = "/mnt/data/cr_json_db"
NEW_JIRA_PATH = "/mnt/data/new_jira.txt"
NEW_JIRA_DUMP_PATH = "/mnt/data/new_jira_dump"
WEAVIATE_URL = "http://localhost:8080"
WEAVIATE_CR_INDEX = "cr_vector_index"
WEAVIATE_DUMP_INDEX = "jira_dump_index"

embedding_model = OpenAIEmbeddings()
llm = ChatOpenAI(model_name="gpt-4")
client = weaviate.Client(url=WEAVIATE_URL)

# --- Tools ---
def create_text_from_json(json_data: dict) -> str:
    summary = json_data.get("summary", "")
    build_info = json_data.get("build_info", "")
    logs_checked = json_data.get("logs_checked", [])
    logs_text = "\n".join(["; ".join(group) for group in logs_checked])
    return f"Summary:\n{summary}\n\nBuild Info:\n{build_info}\n\nLogs Checked:\n{logs_text}"

def index_crs(state):
    if client.schema.exists(WEAVIATE_CR_INDEX):
        client.schema.delete_class(WEAVIATE_CR_INDEX)
    client.schema.create_class({"class": WEAVIATE_CR_INDEX, "vectorizer": "none"})

    store = Weaviate(client, WEAVIATE_CR_INDEX, embedding_model)
    for fname in os.listdir(CR_JSON_FOLDER):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(CR_JSON_FOLDER, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
            cr_number = data.get("cr_number", fname[:-5])
            full_text = create_text_from_json(data)
            store.add_texts([full_text], metadatas=[{"cr_number": cr_number}])
    return {"status": "indexed_crs"}

def index_jira_dump(state):
    if client.schema.exists(WEAVIATE_DUMP_INDEX):
        client.schema.delete_class(WEAVIATE_DUMP_INDEX)
    client.schema.create_class({"class": WEAVIATE_DUMP_INDEX, "vectorizer": "none"})
    store = Weaviate(client, WEAVIATE_DUMP_INDEX, embedding_model)
    for root, _, files in os.walk(NEW_JIRA_DUMP_PATH):
        for file in files:
            fpath = os.path.join(root, file)
            try:
                with open(fpath, "r", encoding="utf-8", errors='ignore') as f:
                    content = f.read()[:2000]
                    store.add_texts([content], metadatas=[{"filename": file}])
            except: pass
    return {"status": "indexed_jira_dump"}

def read_jira(state):
    with open(NEW_JIRA_PATH, "r", encoding="utf-8") as f:
        return {"jira_text": f.read()}

def find_similar_crs(state):
    jira_text = state["jira_text"]
    store = Weaviate(client, WEAVIATE_CR_INDEX, embedding_model)
    results = store.similarity_search(jira_text, k=5)
    return {"similar_crs": results, "jira_text": jira_text}

def compare_crs(state):
    jira_text = state["jira_text"]
    similar_crs = state["similar_crs"]
    store = Weaviate(client, WEAVIATE_DUMP_INDEX, embedding_model)
    report = []
    for cr in similar_crs:
        snippets = cr.page_content.split("\n\n")[:2]
        matches = []
        for snippet in snippets:
            matches.extend(store.similarity_search(snippet, k=2))
        logs_combined = "\n\n".join(f"{m.metadata['filename']}: {m.page_content[:500]}" for m in matches)
        prompt = f"""
        CR Info:
        {cr.page_content}

        JIRA Text:
        {jira_text}

        Logs from new crash:
        {logs_combined}

        Are the issues the same? Should we reuse CR {cr.metadata['cr_number']}?
        """
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a crash diagnosis assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        decision = response['choices'][0]['message']['content']
        report.append(f"CR {cr.metadata['cr_number']}: {decision}")
    return {"comparison_report": "\n\n".join(report)}

# --- LangGraph Setup ---
workflow = StateGraph()
workflow.add_node("index_crs", index_crs)
workflow.add_node("index_jira", index_jira_dump)
workflow.add_node("read_jira", read_jira)
workflow.add_node("find_similar", find_similar_crs)
workflow.add_node("compare", compare_crs)

workflow.set_entry_point("index_crs")
workflow.add_edge("index_crs", "index_jira")
workflow.add_edge("index_jira", "read_jira")
workflow.add_edge("read_jira", "find_similar")
workflow.add_edge("find_similar", "compare")
workflow.add_edge("compare", END)

app = workflow.compile()

if __name__ == "__main__":
    final = app.invoke({})
    print("\n\n=== Final Comparison Report ===\n")
    print(final["comparison_report"])
