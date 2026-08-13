from fastapi import FastAPI,Request
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage,HumanMessage,SystemMessage
import os
from routes import chat
from routes.connectors import gmail

app = FastAPI()

app.include_router(chat.router,prefix='/alexa')
app.include_router(gmail.router,prefix='/api/connectors')

llm= ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
    max_tokens=300
)

class Alexa(BaseModel):
    name: str
    description: str | None = None
    price: float
    tax: float | None = None



@app.get("/cronjob")
def cronjob():
    return {"message":"Server is healthy"}