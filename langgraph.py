from langgraph.graph import StateGraph, END
from langchain.agents import Tool
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import AgentTokenBufferMemory
from langchain.agents import OpenAIFunctionsAgent, AgentExecutor
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Weaviate

import os
import weaviate
import openai
from typing import Dict, List

# --- CONFIG ---
openai.api_key = "YOUR_OPENAI_API_KEY"
CR_FOLDER = "/mnt/data/cr_db"
NEW_JIRA_PATH = "/mnt/data/new_jira.txt"
NEW_JIRA_DUMP_PATH = "/mnt/data/new_jira_dump"
WEAVIATE_URL = "http://localhost:8080"
WEAVIATE_CR_INDEX = "cr_vector_index"
WEAVIATE_DUMP_INDEX = "jira_dump_index"

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-4")
client = weaviate.Client(url=WEAVIATE_URL)

# --- Tools ---
def load_cr_db() -> Dict[str, str]:
    cr_data = {}
    for fname in os.listdir(CR_FOLDER):
        if fname.endswith(".txt"):
            with open(os.path.join(CR_FOLDER, fname), "r", encoding="utf-8") as f:
                cr_data[fname[:-4]] = f.read()
    return cr_data

def index_crs_tool() -> str:
    cr_data = load_cr_db()
    if client.schema.exists(WEAVIATE_CR_INDEX):
        client.schema.delete_class(WEAVIATE_CR_INDEX)
    client.schema.create_class({"class": WEAVIATE_CR_INDEX, "vectorizer": "none"})
    store = Weaviate(client, WEAVIATE_CR_INDEX, embedding_model)
    for cr_number, text in cr_data.items():
        store.add_texts([text], metadatas=[{"cr_number": cr_number}])
    return f"Indexed {len(cr_data)} CRs"

def read_jira_text() -> str:
    with open(NEW_JIRA_PATH, "r", encoding="utf-8") as f:
        return f.read()

def search_similar_crs(jira_text: str, top_k=5):
    store = Weaviate(client, WEAVIATE_CR_INDEX, embedding_model)
    return store.similarity_search(jira_text, k=top_k)

def index_jira_dump() -> str:
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
    return "Indexed JIRA dump logs"

def search_log_matches(snippet: str, top_k=2):
    store = Weaviate(client, WEAVIATE_DUMP_INDEX, embedding_model)
    return store.similarity_search(snippet, k=top_k)

def compare_crs_tool(similar_crs, jira_text):
    report = []
    for cr in similar_crs:
        snippets = cr.page_content.split('\n\n')[:2]
        matches = []
        for snippet in snippets:
            matches.extend(search_log_matches(snippet))
        logs_combined = "\n\n".join(f"{log.metadata['filename']} log:\n{log.metadata['filename'][:1000]}" for log in matches)
        prompt = f"""
        Past CR:
        {cr.page_content}

        New JIRA:
        {jira_text}

        Logs from new crash:
        {logs_combined}

        Are the issues likely the same? Should we use the same CR?
        """
        res = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a crash diagnosis expert."},
                {"role": "user", "content": prompt}
            ]
        )
        report.append(f"CR {cr.metadata['cr_number']}: {res['choices'][0]['message']['content'].strip()}")
    return "\n\n".join(report)

# --- LangGraph ---
def build_graph():
    builder = StateGraph()

    builder.add_node("IndexCRs", RunnableLambda(lambda _: index_crs_tool()))
    builder.add_node("IndexDump", RunnableLambda(lambda _: index_jira_dump()))
    builder.add_node("GetJiraText", RunnableLambda(lambda _: read_jira_text()))
    builder.add_node("FindCRs", RunnableLambda(lambda state: search_similar_crs(state['jira_text'])))
    builder.add_node("Compare", RunnableLambda(lambda state: compare_crs_tool(state['similar_crs'], state['jira_text'])))

    builder.set_entry_point("IndexCRs")
    builder.add_edge("IndexCRs", "IndexDump")
    builder.add_edge("IndexDump", "GetJiraText")
    builder.add_edge("GetJiraText", "FindCRs")
    builder.add_edge("FindCRs", "Compare")
    builder.add_edge("Compare", END)

    builder.set_state_schema({})

    return builder.compile()

if __name__ == "__main__":
    graph = build_graph()
    result = graph.invoke({})
    print(result)
