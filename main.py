from fastapi import FastAPI,Request
from pydantic import BaseModel
from langchain_groq import ChatGroq
import os

app = FastAPI()
llm= ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
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
        intent_name =await body["request"]["intent"]["name"]

        if intent_name == "ChatIntent":
            query =await body["request"]["intent"]["value"]
            reply = llm.invoke(query).content
            print("""query is {query} and reply is {reply}""")
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