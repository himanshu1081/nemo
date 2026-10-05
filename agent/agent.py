from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command

from agent.tools import TOOLS
from services.llm import llm

SYSTEM_PROMPT = SystemMessage(content=(
    "You are Nemo, an AI assistant running on Alexa. Your reply is spoken aloud, "
    "so use plain conversational sentences: no markdown, lists, emojis, URLs or email ids. "
    "Keep replies short. Use the available tools whenever the user asks about their emails; "
    "if a request has several parts, call every tool needed. "
    "When listing emails, mention sender name and subject briefly, and summarize email bodies "
    "instead of reading them word for word. "
    "To send an email, look up the recipient with find_contact unless the user gave a full address, "
    "then call send_email right away. Never ask the user whether the draft is okay or whether to send it: "
    "send_email itself reads the draft back and asks them to confirm. "
    "Never say an email was sent unless send_email returned that it was sent. "
    "If the recipient is ambiguous or you don't know what to write, ask the user first."
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


def confirmation_prompt(draft: dict) -> str:
    return (
        f"Here's the email to {draft['to']}. Subject: {draft['subject']}. "
        f"It says: {draft['body']} Should I send it?"
    )


async def run_agent(text: str, user_id: str, session_id: str) -> str:
    config = {
        "configurable": {"thread_id": session_id, "user_id": user_id},
        "recursion_limit": 8,
    }

    # A pending send_email confirmation gets this utterance as its answer
    state = await graph.aget_state(config)
    if state.interrupts:
        graph_input = Command(resume=text)
    else:
        graph_input = {"messages": [HumanMessage(content=text)]}

    result = await graph.ainvoke(graph_input, config=config)

    if result.get("__interrupt__"):
        return confirmation_prompt(result["__interrupt__"][0].value)
    return result["messages"][-1].content
