from langchain.agents import create_openai_functions_agent, AgentExecutor, tool
from langchain_community.chat_models import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableMap
from langchain_community.vectorstores import Chroma
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.embeddings import Embeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langgraph.graph import END, StateGraph
from typing import TypedDict, Annotated, Sequence
from operator import itemgetter
import os

# --- Define a sample tool ---
@tool
def parse_file(file_path: str) -> str:
    """Parses a file and returns its content. Example: file_path='data/wow.txt'"""
    try:
        with open(file_path, 'r') as f:
            return f.read()
    except Exception as e:
        return f"Error: {str(e)}"

# --- Define the tools list ---
tools = [parse_file]

# --- Create the agent ---
llm = ChatOpenAI(model="gpt-4")
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that can use tools to answer user queries."),
    MessagesPlaceholder("chat_history"),
    ("user", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# --- Wrap agent with message history ---
agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    lambda session_id: memory,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# --- Define graph state ---
class GraphState(TypedDict):
    input: str
    context: str
    agent_output: str
    chat_history: list

# --- Prompt rewriting ---
def rewrite_prompt(state):
    return {"input": f"Rewrite this query for better tool usage: {state['input']}"}

# --- Context retrieval (ChromaDB) ---
embedding = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-base-en-v1.5")
vectorstore = Chroma(persist_directory="chroma", embedding_function=embedding)
retriever = vectorstore.as_retriever()

retrieve_context = (
    itemgetter("input")
    | retriever
    | RunnableLambda(lambda docs: {"context": "\n".join([doc.page_content for doc in docs])})
)

# --- Combine context with input ---
combine_context = RunnableLambda(
    lambda state: {
        "input": f"Use this context to answer the query: {state['input']}\nContext: {state['context']}"
    }
)

# --- Agent execution ---
run_agent = RunnableLambda(lambda state: {"agent_output": agent_with_chat_history.invoke(state)})

# --- Build LangGraph ---
graph_builder = StateGraph(GraphState)
graph_builder.add_node("rewrite_prompt", RunnableLambda(rewrite_prompt))
graph_builder.add_node("retrieve_context", retrieve_context)
graph_builder.add_node("combine_context", combine_context)
graph_builder.add_node("agent_executor", run_agent)

graph_builder.set_entry_point("rewrite_prompt")
graph_builder.add_edge("rewrite_prompt", "retrieve_context")
graph_builder.add_edge("retrieve_context", "combine_context")
graph_builder.add_edge("combine_context", "agent_executor")
graph_builder.add_edge("agent_executor", END)

graph = graph_builder.compile()
