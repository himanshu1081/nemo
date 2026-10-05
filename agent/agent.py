from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command

from agent.memory import TurnContext, build_context_prompt, build_history, load_turn_context, save_turn
from agent.tools import TOOLS
from services.llm import FALLBACK_ERRORS, llm, small_llm

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
    "If the recipient is ambiguous or you don't know what to write, ask the user first. "
    "You can only take actions through your tools. Never claim you did something no tool did, "
    "such as controlling devices, setting routines or reminders; say you can't do that yet. "
    "You may be given what you know about the user and summaries of past conversations; use them "
    "naturally when relevant, and say so plainly if you don't remember something."
))
TOOL_RESULT_CHARS = 600

llm_with_tools = llm.bind_tools(TOOLS).with_fallbacks(
    [small_llm.bind_tools(TOOLS)],
    exceptions_to_handle=FALLBACK_ERRORS,
)


def call_ai(state: MessagesState, config):
    # graph state holds only the current turn; history and memory come from Supabase
    context: TurnContext = config["configurable"].get("memory_context") or TurnContext()
    context_prompt = build_context_prompt(context)
    system = SystemMessage(content=SYSTEM_PROMPT.content + ("\n\n" + context_prompt if context_prompt else ""))
    response = llm_with_tools.invoke([system] + build_history(context) + state["messages"])
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


@dataclass
class AgentReply:
    text: str
    context: TurnContext


def _tool_results(messages) -> list[dict]:
    return [
        {"tool": m.name, "result": str(m.content)[:TOOL_RESULT_CHARS]}
        for m in messages if isinstance(m, ToolMessage)
    ]


async def run_agent(text: str, alexa_user_id: str, session_id: str,
                    request_id: str | None, gmail_user_id: str | None) -> AgentReply:
    # The checkpointer only carries the current turn, so a pending send_email confirmation survives until the next one
    thread = {"configurable": {"thread_id": session_id}}
    state = await graph.aget_state(thread)
    resuming = bool(state.interrupts)
    if not resuming:
        await graph.checkpointer.adelete_thread(session_id)

    # "yes" to a confirmation doesn't need past-conversation retrieval
    context = await load_turn_context(alexa_user_id, session_id, text, use_rag=not resuming)

    config = {
        "configurable": {"thread_id": session_id, "user_id": gmail_user_id, "memory_context": context},
        "recursion_limit": 8,
    }
    graph_input = Command(resume=text) if resuming else {"messages": [HumanMessage(content=text)]}
    result = await graph.ainvoke(graph_input, config=config)

    if result.get("__interrupt__"):
        reply = confirmation_prompt(result["__interrupt__"][0].value)
    else:
        reply = result["messages"][-1].content

    await save_turn(context, request_id, text, reply, _tool_results(result["messages"]))
    return AgentReply(reply, context)


async def end_session(session_id: str) -> None:
    await graph.checkpointer.adelete_thread(session_id)
