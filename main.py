from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    print("Server hit");
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

@app.get("/cronjob")
def cronjob():
    return {"message":"Server is healthy"}