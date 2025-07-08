LangGraph and LangChain-based CR Matching Pipeline (with FAISS instead of Qdrant)

from langgraph.graph import END, StateGraph from langchain.chat_models import ChatOpenAI from langchain.embeddings import OpenAIEmbeddings from langchain.vectorstores import FAISS from langchain.agents import initialize_agent, Tool

import os import json import openai from rapidfuzz import fuzz from sklearn.metrics.pairwise import cosine_similarity import numpy as np import faiss import pickle

--- CONFIG ---

openai.api_key = "YOUR_OPENAI_API_KEY" CR_JSON_FOLDER = "/mnt/data/cr_json_db" NEW_JIRA_PATH = "/mnt/data/new_jira.txt" NEW_JIRA_DUMP_PATH = "/mnt/data/new_jira_dump" FAISS_INDEX_FILE = "cr_faiss_index.pkl"

embedding_model = OpenAIEmbeddings() llm = ChatOpenAI(model_name="gpt-4")

--- Tools for the Log Explorer Agent ---

def get_embedding(text): return embedding_model.embed_query(text)

def semantic_score(log_snippet, line): try: v1 = get_embedding(log_snippet) v2 = get_embedding(line) return cosine_similarity([v1], [v2])[0][0] except: return 0.0

def fuzzy_score(a, b): return fuzz.partial_ratio(a, b) / 100.0

def match_log_snippet_in_dump(snippet): best_match = ("", 0, "") for root, _, files in os.walk(NEW_JIRA_DUMP_PATH): for file in files: fpath = os.path.join(root, file) try: with open(fpath, "r", encoding="utf-8", errors='ignore') as f: lines = f.readlines()[-100:] for line in lines: sem = semantic_score(snippet, line) fuzz_s = fuzzy_score(snippet, line) score = max(sem, fuzz_s) if score > best_match[1]: best_match = (line.strip(), score, file) except: continue return f"Match in {best_match[2]} [Score: {best_match[1]:.2f}]:\n{best_match[0]}"

def extract_register_value(register_name: str): matches = [] for root, _, files in os.walk(NEW_JIRA_DUMP_PATH): for file in files: fpath = os.path.join(root, file) try: with open(fpath, "r", encoding="utf-8", errors='ignore') as f: for line in f: if register_name.lower() in line.lower(): matches.append(f"{file}: {line.strip()}") except: continue if not matches: return f"No register values found for '{register_name}'" return f"Register Matches for '{register_name}':\n" + "\n".join(matches)

tools = [ Tool(name="MatchLogSnippetInDump", func=match_log_snippet_in_dump, description="Find similar log in the new crash dump"), Tool(name="ExtractRegisterValue", func=extract_register_value, description="Extract values of a register from dump") ]

log_explorer_agent = initialize_agent(tools, llm, agent_type="zero-shot-react-description", verbose=True)

--- FAISS Indexing ---

def index_crs(): texts = [] metadatas = [] embeddings = []

for fname in os.listdir(CR_JSON_FOLDER):
    if not fname.endswith(".json"):
        continue
    fpath = os.path.join(CR_JSON_FOLDER, fname)
    with open(fpath, "r", encoding="utf-8") as f:
        data = json.load(f)
        cr_number = data.get("cr_number", fname[:-5])
        summary = data.get("summary", "")
        crash_signature = data.get("crash_signature", "")
        index_text = f"{crash_signature}\n{summary}" if crash_signature else summary
        texts.append(index_text)
        embeddings.append(get_embedding(index_text))
        metadatas.append({"cr_number": cr_number, "full_data": data})

index = faiss.IndexFlatL2(len(embeddings[0]))
index.add(np.array(embeddings).astype("float32"))

with open(FAISS_INDEX_FILE, "wb") as f:
    pickle.dump((index, texts, metadatas), f)

--- LangGraph Nodes ---

def read_jira(state): with open(NEW_JIRA_PATH, "r", encoding="utf-8") as f: return {"jira_text": f.read()}

def find_similar_crs(state): jira_text = state["jira_text"] query_vector = np.array([get_embedding(jira_text)]).astype("float32")

with open(FAISS_INDEX_FILE, "rb") as f:
    index, texts, metadatas = pickle.load(f)

distances, indices = index.search(query_vector, 5)
results = []
for idx in indices[0]:
    results.append(type("FAISSResult", (), {"metadata": metadatas[idx], "page_content": texts[idx]}))

return {"similar_crs": results, "jira_text": jira_text}

def compare_crs_with_jira(state): similar_crs = state["similar_crs"] jira_text = state["jira_text"] report = []

for cr in similar_crs:
    cr_number = cr.metadata.get("cr_number")
    full_data = cr.metadata.get("full_data", {})
    logs_checked = full_data.get("logs_checked", [])

    matched_log_responses = []
    for group in logs_checked:
        for snippet in group:
            response = log_explorer_agent.run(
                f"Search for log or value similar to: '{snippet}' in the new crash dump"
            )
            matched_log_responses.append(response)

    final_prompt = f"""
    CR {cr_number} Summary:
    {cr.page_content}

    CR {cr_number} Callstack (if any):
    {full_data.get('callstack', 'N/A')}

    JIRA Summary and Callstack:
    {jira_text}
    {jira_text}

    Matched Logs from New Dump:
    {'\n\n'.join(matched_log_responses)}

    Question: Based on the summaries and matched logs, is this JIRA likely a duplicate of CR {cr_number}? Explain.
    """
    decision = llm.predict(final_prompt)
    report.append(f"CR {cr_number} Comparison:\n{decision}\n")

return {"comparison_report": "\n\n".join(report)}

--- LangGraph Setup ---

workflow = StateGraph() workflow.add_node("read_jira", read_jira) workflow.add_node("find_similar", find_similar_crs) workflow.add_node("compare", compare_crs_with_jira)

workflow.set_entry_point("read_jira") workflow.add_edge("read_jira", "find_similar") workflow.add_edge("find_similar", "compare") workflow.add_edge("compare", END)

app = workflow.compile()

if name == "main": import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--index", action="store_true", help="Index CRs into FAISS")
args = parser.parse_args()

if args.index:
    index_crs()
    print("CRs indexed successfully.")
else:
    final = app.invoke({})
    print("\n\n=== Final Comparison Report ===\n")
    print(final["comparison_report"])

