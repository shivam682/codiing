LangGraph and LangChain-based CR Matching Pipeline (with Agents for Dynamic Tasks)

from langgraph.graph import END, StateGraph from langchain.chat_models import ChatOpenAI from langchain.embeddings import OpenAIEmbeddings from langchain_community.vectorstores import Qdrant from langchain.vectorstores.qdrant import Qdrant as QdrantStore from langchain.agents import initialize_agent, Tool

import os import json import openai from rapidfuzz import fuzz from sklearn.metrics.pairwise import cosine_similarity import numpy as np from qdrant_client import QdrantClient from qdrant_client.http.models import Distance, VectorParams

--- CONFIG ---

openai.api_key = "YOUR_OPENAI_API_KEY" CR_JSON_FOLDER = "/mnt/data/cr_json_db" NEW_JIRA_PATH = "/mnt/data/new_jira.txt" NEW_JIRA_DUMP_PATH = "/mnt/data/new_jira_dump" QDRANT_URL = "http://localhost:6333" QDRANT_COLLECTION = "cr_vector_index"

embedding_model = OpenAIEmbeddings() llm = ChatOpenAI(model_name="gpt-4") qdrant_client = QdrantClient(url=QDRANT_URL)

--- Tools for the Log Explorer Agent ---

def get_embedding(text): return embedding_model.embed_query(text)

def semantic_score(log_snippet, line): try: v1 = get_embedding(log_snippet) v2 = get_embedding(line) return cosine_similarity([v1], [v2])[0][0] except: return 0.0

def fuzzy_score(a, b): return fuzz.partial_ratio(a, b) / 100.0

def match_log_snippet_in_dump(snippet): best_match = ("", 0, "")  # content, score, filename for root, _, files in os.walk(NEW_JIRA_DUMP_PATH): for file in files: fpath = os.path.join(root, file) try: with open(fpath, "r", encoding="utf-8", errors='ignore') as f: content = f.read() sem = semantic_score(snippet, content) fuzz_s = fuzzy_score(snippet, content) score = max(sem, fuzz_s) if score > best_match[1]: best_match = (content.strip(), score, file) except: continue return f"Match in {best_match[2]} [Score: {best_match[1]:.2f}]:\n{best_match[0]}", best_match[1]

def extract_register_value(register_name: str): matches = [] for root, _, files in os.walk(NEW_JIRA_DUMP_PATH): for file in files: fpath = os.path.join(root, file) try: with open(fpath, "r", encoding="utf-8", errors='ignore') as f: for line in f: if register_name.lower() in line.lower(): matches.append(f"{file}: {line.strip()}") except: continue if not matches: return f"No register values found for '{register_name}'" return f"Register Matches for '{register_name}':\n" + "\n".join(matches)

tools = [ Tool(name="MatchLogSnippetInDump", func=lambda snippet: match_log_snippet_in_dump(snippet)[0], description="Find similar log in the new crash dump"), Tool(name="ExtractRegisterValue", func=extract_register_value, description="Extract values of a register from dump") ]

log_explorer_agent = initialize_agent(tools, llm, agent_type="zero-shot-react-description", verbose=True)

--- Optional: Index CRs Function ---

def index_crs(): qdrant_client.recreate_collection( collection_name=QDRANT_COLLECTION, vectors_config=VectorParams(size=1536, distance=Distance.COSINE), )

texts = []
metadatas = []
for fname in os.listdir(CR_JSON_FOLDER):
    if not fname.endswith(".json"):
        continue
    fpath = os.path.join(CR_JSON_FOLDER, fname)
    with open(fpath, "r", encoding="utf-8") as f:
        data = json.load(f)
        cr_number = data.get("cr_number", fname[:-5])
        summary = data.get("summary", "")
        crash_signature = data.get("crash_signature", "")
        callstack = data.get("callstack", "")
        index_text = f"{crash_signature}\n{summary}\n{callstack}".strip()
        texts.append(index_text)
        metadatas.append({"cr_number": cr_number, "full_data": data})

store = QdrantStore(
    client=qdrant_client,
    collection_name=QDRANT_COLLECTION,
    embedding=embedding_model,
)
store.add_texts(texts, metadatas=metadatas)

--- LangGraph Nodes ---

def read_jira(state): with open(NEW_JIRA_PATH, "r", encoding="utf-8") as f: return {"jira_text": f.read()}

def find_similar_crs(state): jira_text = state["jira_text"] store = QdrantStore( client=qdrant_client, collection_name=QDRANT_COLLECTION, embedding=embedding_model ) results = store.similarity_search(jira_text, k=5) return {"similar_crs": results, "jira_text": jira_text}

def compare_crs_with_jira(state): similar_crs = state["similar_crs"] jira_text = state["jira_text"] os.makedirs("cr_analysis_reports", exist_ok=True)

for cr in similar_crs:
    cr_number = cr.metadata.get("cr_number")
    full_data = cr.metadata.get("full_data", {})
    logs_checked = full_data.get("logs_checked", [])

    matched_log_responses = []
    for group in logs_checked:
        for snippet in group:
            response, _ = match_log_snippet_in_dump(snippet)
            matched_log_responses.append(response)

    matched_logs = "\n\n".join(matched_log_responses)
    cr_summary = full_data.get("summary", "")
    cr_callstack = full_data.get("callstack", "")

    prompt = {
        "system": "You are an expert in crash analysis. Analyze whether a new crash matches a past CR by comparing the new crash logs against the crash signature, summary, and callstack of the CR.",
        "input": f"""

We are analyzing a new crash report (JIRA) and trying to determine whether it is similar to an existing Crash Report (CR).

Here is the information from the existing CR:

CR Number: {cr_number}

Summary: {cr_summary}

Callstack: {cr_callstack or 'N/A'}


Based on this context, analyze the following logs and summary of the new crash:

--- New Crash Logs --- {matched_logs}

--- New Crash JIRA Summary --- {jira_text}

Please explain in detail whether this new crash log and description is related to the CR. Do not score it. Just explain reasoning clearly. Only return the analysis. """ }

analysis = llm.invoke(prompt)
    with open(f"cr_analysis_reports/{cr_number}.txt", "w", encoding="utf-8") as f:
        f.write(analysis)

return {"status": "analysis_generated"}

--- LangGraph Setup ---

workflow = StateGraph() workflow.add_node("read_jira", read_jira) workflow.add_node("find_similar", find_similar_crs) workflow.add_node("compare", compare_crs_with_jira)

workflow.set_entry_point("read_jira") workflow.add_edge("read_jira", "find_similar") workflow.add_edge("find_similar", "compare") workflow.add_edge("compare", END)

app = workflow.compile()

if name == "main": import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--index", action="store_true", help="Index CRs into Qdrant")
parser.add_argument("--score", action="store_true", help="Score CRs after analysis")
args = parser.parse_args()

if args.index:
    index_crs()
    print("CRs indexed successfully.")
elif args.score:
    result = generate_final_json_score()
    print("Similarity result:\n", result)
else:
    final = app.invoke({})
    print("Analysis files generated. Run with `--score` to evaluate them.")

