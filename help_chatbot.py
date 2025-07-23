from langchain.chat_models import ChatOpenAI
from langchain.agents import Tool, ZeroShotAgent, AgentExecutor
from langchain.prompts import PromptTemplate
from langchain.tools import tool
from langgraph.graph import StateGraph, END
from typing import TypedDict

# ----------------------------
# 1. Define tools
# ----------------------------
@tool
def read_file(path: str) -> str:
    """Read content from a file at given path"""
    with open(path, 'r') as f:
        return f.read()

@tool
def count_words(text: str) -> int:
    """Count number of words in given text"""
    return len(text.split())

tools = [read_file, count_words]

# ----------------------------
# 2. Create ZeroShotAgent manually
# ----------------------------
PREFIX = """You are a helpful assistant that can read files and count words.
You can use these tools:
- read_file(path): Reads a file and returns its content.
- count_words(text): Returns number of words in the text."""

FORMAT_INSTRUCTIONS = """Use this format:
Question: ...
Thought: ...
Action: ...
Action Input: ...
Observation: ...
... (repeat)
Thought: I now know the final answer
Final Answer: ..."""

SUFFIX = """Begin!

Question: {input}
{agent_scratchpad}"""

llm = ChatOpenAI(temperature=0)

prompt = ZeroShotAgent.create_prompt(
    tools=tools,
    prefix=PREFIX,
    suffix=SUFFIX,
    format_instructions=FORMAT_INSTRUCTIONS,
    input_variables=["input", "agent_scratchpad"]
)

agent = ZeroShotAgent(llm_chain=prompt | llm, tools=tools)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# ----------------------------
# 3. Define LangGraph State
# ----------------------------
class AgentState(TypedDict):
    input: str
    output: str

# ----------------------------
# 4. Agent node
# ----------------------------
def run_agent(state: AgentState) -> AgentState:
    output = agent_executor.run(state["input"])
    return {"input": state["input"], "output": output}

# ----------------------------
# 5. Conditional edge
# ----------------------------
def should_continue(state: AgentState) -> str:
    if "bye" in state["input"].lower():
        return "end"
    return "continue"

# ----------------------------
# 6. Final node (optional)
# ----------------------------
def final_node(state: AgentState) -> AgentState:
    print("Conversation ended.")
    print("Final output:", state["output"])
    return state

# ----------------------------
# 7. Build LangGraph flow
# ----------------------------
graph = StateGraph(AgentState)
graph.add_node("agent", run_agent)
graph.add_node("final", final_node)
graph.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "agent",
        "end": "final"
    }
)
graph.set_entry_point("agent")
graph.set_finish_point("final")
chat_graph = graph.compile()

# ----------------------------
# 8. Run chat loop
# ----------------------------
if __name__ == "__main__":
    print("🤖 LangGraph with ZeroShotAgent — Type 'bye' to exit\n")
    while True:
        query = input("🧑 You: ")
        result = chat_graph.invoke({"input": query})
        print("🤖 Agent:", result["output"])
        if "bye" in query.lower():
            break
