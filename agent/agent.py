from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agent.tools import TOOLS
from services.llm import llm

SYSTEM_PROMPT = SystemMessage(content=(
    "You are Nemo, an AI assistant running on Alexa. Your reply is spoken aloud, "
    "so use plain conversational sentences: no markdown, lists, emojis, URLs or email ids. "
    "Keep replies short. Use the available tools whenever the user asks about their emails; "
    "if a request has several parts, call every tool needed. "
    "When listing emails, mention sender name and subject briefly, and summarize email bodies "
    "instead of reading them word for word."
))

llm_with_tools = llm.bind_tools(TOOLS)


def call_ai(state: MessagesState):
    response = llm_with_tools.invoke([SYSTEM_PROMPT] + state["messages"])
    return {"messages": [response]}


builder = StateGraph(MessagesState)
builder.add_node("agent", call_ai)
builder.add_node("tools", ToolNode(TOOLS))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")

graph = builder.compile(checkpointer=MemorySaver())


async def run_agent(text: str, user_id: str, session_id: str) -> str:
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=text)]},
        config={
            "configurable": {"thread_id": session_id, "user_id": user_id},
            "recursion_limit": 8,
        },
    )
    return result["messages"][-1].content
