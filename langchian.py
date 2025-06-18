from langchain.agents import Tool, AgentExecutor, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Weaviate

import os
import weaviate
import openai

# --- CONFIG ---
openai.api_key = "YOUR_OPENAI_API_KEY"
CR_FOLDER = "/mnt/data/cr_db"
NEW_JIRA_PATH = "/mnt/data/new_jira.txt"
NEW_JIRA_DUMP_PATH = "/mnt/data/new_jira_dump"
WEAVIATE_URL = "http://localhost:8080"
WEAVIATE_CR_INDEX = "cr_vector_index"
WEAVIATE_DUMP_INDEX = "jira_dump_index"

embedding_model = OpenAIEmbeddings()
llm = ChatOpenAI(model_name="gpt-4")
client = weaviate.Client(url=WEAVIATE_URL)

# --- Tools ---
def load_cr_db():
    cr_data = {}
    for fname in os.listdir(CR_FOLDER):
        if fname.endswith(".txt"):
            with open(os.path.join(CR_FOLDER, fname), "r", encoding="utf-8") as f:
                cr_data[fname[:-4]] = f.read()
    return cr_data

def index_crs():
    cr_data = load_cr_db()
    if client.schema.exists(WEAVIATE_CR_INDEX):
        client.schema.delete_class(WEAVIATE_CR_INDEX)
    client.schema.create_class({"class": WEAVIATE_CR_INDEX, "vectorizer": "none"})
    store = Weaviate(client, WEAVIATE_CR_INDEX, embedding_model)
    for cr_number, text in cr_data.items():
        store.add_texts([text], metadatas=[{"cr_number": cr_number}])
    return f"Indexed {len(cr_data)} CRs"

def read_jira():
    with open(NEW_JIRA_PATH, "r", encoding="utf-8") as f:
        return f.read()

def index_jira_dump():
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

def find_similar_crs(jira_text: str):
    store = Weaviate(client, WEAVIATE_CR_INDEX, embedding_model)
    return store.similarity_search(jira_text, k=5)

def search_logs(snippet: str):
    store = Weaviate(client, WEAVIATE_DUMP_INDEX, embedding_model)
    return store.similarity_search(snippet, k=2)

def compare_crs_tool(jira_text: str):
    similar_crs = find_similar_crs(jira_text)
    report = []
    for cr in similar_crs:
        snippets = cr.page_content.split("\n\n")[:2]
        matches = []
        for snippet in snippets:
            matches.extend(search_logs(snippet))
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
    return "\n\n".join(report)

# --- Setup LangChain Agent ---
tools = [
    Tool(name="Index CRs", func=lambda _: index_crs(), description="Index historical CRs"),
    Tool(name="Index JIRA Dump", func=lambda _: index_jira_dump(), description="Index JIRA dump logs"),
    Tool(name="Read JIRA Text", func=lambda _: read_jira(), description="Read new JIRA text"),
    Tool(name="Compare CRs", func=lambda _: compare_crs_tool(read_jira()), description="Compare JIRA with existing CRs")
]

memory = ConversationBufferMemory(memory_key="chat_history")
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    memory=memory,
    verbose=True
)

if __name__ == "__main__":
    print(agent.run("Index CRs"))
    print(agent.run("Index JIRA Dump"))
    print(agent.run("Compare CRs"))
