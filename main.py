from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    print("Server hit");
    return {"message": "Hello vexa this side"}