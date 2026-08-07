from fastapi import FastAPI,Request
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage,HumanMessage,SystemMessage
import os

app = FastAPI()
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

@app.get("/")
def home():
    print("Server hit")
    return {"message":"Server is healthy"}



@app.post("/alexa")
async def alexa(request: Request):
    conversation_history=[]

    print("Server hit on alexa skill")

    body = await request.json()

    request_type = body["request"]["type"]

    if request_type == "LaunchRequest":
        return {
            "version": "1.0",
            "response": {
                "outputSpeech": {
                    "type": "PlainText",
                    "text": "Hello! I'm Vexa. What would you like to know?"
                },
                "shouldEndSession": False
            }
        }

    elif request_type == "IntentRequest":
        intent_name =body["request"]["intent"]["name"]

        if intent_name == "ChatIntent":
            query = body["request"]["intent"]["slots"]["message"]["value"]
            conversation_history.append(HumanMessage(content=query))

            systemPrompt = SystemMessage(content="You are Vexa AI running on Alexa. User has asked you question and you are supposed to answer them plus use appropriate tools possible. Keep replies short")

            reply =llm.invoke([systemPrompt]+conversation_history).content

            conversation_history.append(reply)

            return {
                "version": "1.0",
                "response": {
                    "outputSpeech": {
                        "type": "PlainText",
                        "text": reply
                    },
                    "shouldEndSession": False
                }
            }


@app.get("/cronjob")
def cronjob():
    return {"message":"Server is healthy"}