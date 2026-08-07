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
def alexa(request:Request):
    print("Server hit on alexa skill");
    body = request.json()
    userIntent = body.request.type
    if(userIntent == "LaunchRequest"):
        return {
            "version": "1.0",
            "response": {
                "outputSpeech": {
                    "type": "PlainText",
                    "text": "Hello Vexa this side"
                },
                "shouldEndSession": False
            }
        }

    if(userIntent=="ChatIntent"):
        return {
            "version": "1.0",
            "response": {
                "outputSpeech": {
                    "type": "PlainText",
                    "text": "Hello let's talk"
                },
                "shouldEndSession": False
            }
        }


@app.get("/cronjob")
def cronjob():
    return {"message":"Server is healthy"}