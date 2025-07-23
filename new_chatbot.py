from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda, RunnableMap
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain.agents import Tool, AgentExecutor, create_tool_calling_agent
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.memory import ChatMessageHistory
import os

# Define state
class GraphState(TypedDict):
    question: str
    context: str
    retrieved: str
    tool_input: str
    tool_output: str
    answer: str

# LLM setup
llm = ChatOpenAI(model='gpt-4', temperature=0)

# Vector DB
embedding = OllamaEmbeddings(model="nomic-embed-text")
vectordb = Chroma(persist_directory="./chroma_db", embedding_function=embedding)
retriever = vectordb.as_retriever()

# === Prompt Templates === #

rewrite_question_prompt = PromptTemplate.from_template(
    """
You are given a user question. Your job is to rewrite it into a format that can be used to extract requirements or checks for a system state.

Example:
User Question: My system is in wow.
Rewritten: What conditions or requirements are needed for a system to be in 'wow' state?

User Question: {question}
Rewritten:
"""
)

extract_checklist_prompt = PromptTemplate.from_template(
    """
You're given context about a system from retrieved documents. Based on that context, generate a list of explicit checks (in bullet points) that must be performed.

Context:
{context}

Output Format:
- Check if variable X exists in file `abc.txt`
- Ensure log line 'Success' is in `dump.log`
... etc
"""
)

final_answer_prompt = PromptTemplate.from_template(
    """
Given the checklist output from tool calls and the user's original query, synthesize a final answer in clear and concise form.

Checklist Result:
{tool_output}

User Question:
{question}

Answer:
"""
)

# === Nodes as Lambdas === #

def rewrite_question_node(state: GraphState) -> GraphState:
    rewritten = llm.invoke(rewrite_question_prompt.format(question=state['question']))
    return {**state, "question": rewritten.content}

def retrieve_node(state: GraphState) -> GraphState:
    docs = retriever.get_relevant_documents(state['question'])
    context = "\n".join([doc.page_content for doc in docs])
    return {**state, "context": context}

def extract_checklist_node(state: GraphState) -> GraphState:
    checklist = llm.invoke(extract_checklist_prompt.format(context=state['context']))
    return {**state, "tool_input": checklist.content}

def final_answer_node(state: GraphState) -> GraphState:
    answer = llm.invoke(final_answer_prompt.format(
        tool_output=state['tool_output'],
        question=state['question']
    ))
    return {**state, "answer": answer.content}

# === Tools (Mocked for Now) === #
def dummy_tool_executor(input_text: str) -> str:
    return f"[Mocked tool execution on]:\n{input_text}\n(Result: All checks passed)"

dummy_tool = Tool(
    name="checklist_tool",
    func=dummy_tool_executor,
    description="Tool that verifies checklists against dump directory"
)

agent = create_tool_calling_agent(llm, [dummy_tool])
agent_executor = AgentExecutor(agent=agent, tools=[dummy_tool], verbose=True)

def run_tool_node(state: GraphState) -> GraphState:
    output = agent_executor.invoke({"input": state['tool_input']})
    return {**state, "tool_output": output['output']}

# === LangGraph === #
graph_builder = StateGraph(GraphState)
graph_builder.add_node("rewrite_question", RunnableLambda(rewrite_question_node))
graph_builder.add_node("retrieve", RunnableLambda(retrieve_node))
graph_builder.add_node("extract_checklist", RunnableLambda(extract_checklist_node))
graph_builder.add_node("run_tool", RunnableLambda(run_tool_node))
graph_builder.add_node("final_answer", RunnableLambda(final_answer_node))

graph_builder.set_entry_point("rewrite_question")
graph_builder.add_edge("rewrite_question", "retrieve")
graph_builder.add_edge("retrieve", "extract_checklist")
graph_builder.add_edge("extract_checklist", "run_tool")
graph_builder.add_edge("run_tool", "final_answer")
graph_builder.add_edge("final_answer", END)

# === Memory Support === #
chat_history = ChatMessageHistory()
def get_memory(session_id: str):
    return chat_history

graph = graph_builder.compile()
app = RunnableWithMessageHistory(
    graph,
    get_session_history=get_memory,
    input_messages_key="question",
    history_messages_key="messages"
)

# === Chatbot Loop === #
print("\nWelcome to the AI Troubleshooting Assistant. Type 'exit' to quit.\n")

while True:
    user_query = input("You: ")
    if user_query.lower() == "exit":
        break

    response = app.invoke({"question": user_query}, config={"configurable": {"session_id": "shivam-session"}})
    print(f"\nAI: {response['answer']}\n")
