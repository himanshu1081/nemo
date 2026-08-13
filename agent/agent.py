from langgraph.graph import START,END,StateGraph
from typing import TypedDict
from langchain_groq import ChatGroq
from langgraph.prebuilt import ToolNode
from services.resend import resend
class AgentState(TypedDict):
    message:str
    response:str

def call_ai(state:AgentState)->AgentState:
    systemPrompt = SystemMessage(content="You are Nemo AI running on Alexa. User has asked you question and you are supposed to answer them plus use appropriate tools possible. Keep replies short")
    response = llm.invoke([system_message]+state["messages"])    
    return {"messages": [response]}

def should_continue(state:AgentState)->AgentState:
    last_message = state["message"][-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"

