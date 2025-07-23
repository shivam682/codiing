from langchain_core.prompts import PromptTemplate
from langgraph.graph import END, StateGraph
from langchain_core.runnables import RunnableLambda
from langgraph.prebuilt import ToolNode, ToolExecutor, ChatAgentExecutor
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain_core.utils.memory import MemorySaver
from langchain_community.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph.app import MessageGraph

# Tool definition
@tool
def hello_tool(input: str) -> str:
    return f"hello_tool: {input}"

# LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)

# Prompt
prompt = PromptTemplate.from_template("""
You are a helpful AI assistant.

Previous conversation:
{chat_history}

User input: {input}

Respond appropriately.
""")

# Router node to determine question type
def route_node(state):
    messages = state["messages"]
    if len(messages) < 2:
        return "new_question"
    last = messages[-2].content.lower()
    current = messages[-1].content.lower()
    if any(keyword in current for keyword in ["last", "previous", "follow up", "what about", "that", "those"]):
        return "follow_up"
    return "new_question"

# Tool executor
tool_executor = ToolExecutor(tools=[hello_tool])

# Agent executor
agent_executor = ChatAgentExecutor.from_llm_and_tools(llm=llm, tools=[hello_tool])

# Graph
workflow = StateGraph({"messages": list})

# Add router and other nodes
workflow.add_node("router", RunnableLambda(route_node))
workflow.add_node("agent", agent_executor)
workflow.add_node("tool", ToolNode(tools=[hello_tool]))

# Edges based on router decision
workflow.add_edge("router", "agent", condition="new_question")
workflow.add_edge("router", "tool", condition="follow_up")
workflow.add_edge("agent", END)
workflow.add_edge("tool", END)

# Router is entrypoint
workflow.set_entry_point("router")

graph = workflow.compile()

# Memory and message handling
memory = ConversationBufferMemory(return_messages=True)
memory_saver = MemorySaver()
runnable = RunnableWithMessageHistory(graph, memory_saver, input_messages_key="messages", history_messages_key="messages")

# Streaming app
app = MessageGraph(runnable)

@app.stream("messages", config={"stream_mode": "values"})
def chat(message):
    return {"messages": [HumanMessage(content=message)]}

# Run loop (simulate chatbot)
if __name__ == "__main__":
    print("Chatbot ready. Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        for output in chat.invoke(user_input):
            print("AI:", output.content)

