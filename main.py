from fastapi import FastAPI,Request
from pydantic import BaseModel

app = FastAPI()

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
        intent_name = body["request"]["intent"]["name"]

        if intent_name == "ChatIntent":
            return {
                "version": "1.0",
                "response": {
                    "outputSpeech": {
                        "type": "PlainText",
                        "text": "Let's talk!"
                    },
                    "shouldEndSession": False
                }
            }


@app.get("/cronjob")
def cronjob():
    return {"message":"Server is healthy"}