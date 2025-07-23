from typing import List, TypedDict
import os
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableLambda
from langchain.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.memory import ChatMessageHistory

# ---- Define State ----
class State(TypedDict):
    messages: List
    dump_dir: str
    query: str
    retrieved_docs: str
    file_insights: str

# ---- Tools ----
@tool
def read_file(path: str) -> str:
    """Reads a file's content"""
    with open(path, 'r') as f:
        return f.read()

@tool
def list_files(dir_path: str) -> List[str]:
    """Lists all file paths recursively in a directory"""
    result = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            result.append(os.path.join(root, file))
    return result

@tool
def search_file(path: str, keyword: str) -> str:
    """Search for a keyword in a file and return matching lines"""
    results = []
    with open(path, 'r') as f:
        for line in f:
            if keyword in line:
                results.append(line.strip())
    return '\n'.join(results)

file_tools = [read_file, list_files, search_file]

# ---- Vector DB Setup ----
persist_dir = "chroma_db"
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = Chroma(persist_directory=persist_dir, embedding_function=embedding_model)
retriever = vectordb.as_retriever()

# ---- LLM Setup ----
llm = ChatOpenAI(model="gpt-4", temperature=0)
retrieval_qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

# ---- Prompts ----
rewrite_query_prompt = PromptTemplate.from_template(
    """
System: Reframe the user query to explicitly describe the requirement based on system behavior.

Example:
User Query: "System is in WOW"
Rewritten: "What are the possible reasons or configurations that cause a system to enter WOW state?"

User Query: "{query}"
Rewritten:
"""
)

rewrite_context_prompt = PromptTemplate.from_template(
    """
System: You have the following context from retrieved DB:
"""
{context}
"""

Update every file path or filename by prefixing it with base path: {dump_dir}.
Return the updated context only.
"""
)

analyze_file_prompt = PromptTemplate.from_template(
    """
System: Using the context below, extract necessary data by using the tools.

Context:
{context}

Give insights like: "Value of X is Y from file Z" or "No match found for ABC".
"""
)

final_response_prompt = PromptTemplate.from_template(
    """
System: Summarize the answer using all the given data.

Original User Query:
{query}

Context Info:
{context}

File Insights:
{insights}

Response:
"""
)

# ---- Nodes ----
def ask_for_dump_dir(state: State):
    if not state.get("dump_dir"):
        return {"messages": state["messages"] + [AIMessage(content="Please provide the base dump directory for file inspection.")]}
    return state

def rewrite_query(state: State):
    query = state["messages"][-1].content
    rewritten = llm.invoke(rewrite_query_prompt.format(query=query)).content
    return {"query": rewritten, "messages": state["messages"]}

def retrieve_documents(state: State):
    result = retrieval_qa.invoke({"query": state["query"]})
    return {"retrieved_docs": result['result'], "messages": state["messages"]}

def rewrite_context(state: State):
    updated = llm.invoke(rewrite_context_prompt.format(
        context=state["retrieved_docs"],
        dump_dir=state["dump_dir"]
    )).content
    return {"retrieved_docs": updated, "messages": state["messages"]}

tool_node = ToolNode(tools=file_tools, prompt=analyze_file_prompt)

def format_final_response(state: State):
    final = llm.invoke(final_response_prompt.format(
        query=state["query"],
        context=state["retrieved_docs"],
        insights=state["file_insights"]
    )).content
    return {"messages": state["messages"] + [AIMessage(content=final)]}

# ---- LangGraph ----
workflow = StateGraph(State)
workflow.add_node("ask_for_dump_dir", ask_for_dump_dir)
workflow.add_node("rewrite_query", rewrite_query)
workflow.add_node("retrieve_docs", retrieve_documents)
workflow.add_node("rewrite_context", rewrite_context)
workflow.add_node("analyze_files", tool_node)
workflow.add_node("final_response", format_final_response)

workflow.set_entry_point("ask_for_dump_dir")
workflow.add_edge("ask_for_dump_dir", "rewrite_query")
workflow.add_edge("rewrite_query", "retrieve_docs")
workflow.add_edge("retrieve_docs", "rewrite_context")
workflow.add_edge("rewrite_context", "analyze_files")
workflow.add_edge("analyze_files", "final_response")
workflow.add_edge("final_response", END)

# ---- Memory ----
memory = {}
def get_session_history(session_id):
    if session_id not in memory:
        memory[session_id] = ChatMessageHistory()
    return memory[session_id]

runnable = RunnableWithMessageHistory(
    workflow.compile(),
    get_session_history,
    input_messages_key="messages",
    history_messages_key="messages"
)

# ---- CLI Interface ----
if __name__ == "__main__":
    session_id = "shivam"
    dump_dir = input("Enter base dump directory: ")
    print("Type 'exit' to quit.")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break

        result = runnable.invoke(
            {"messages": [HumanMessage(content=user_input)], "dump_dir": dump_dir},
            config={"configurable": {"session_id": session_id}}
        )
        print("Bot:", result["messages"][-1].content)
