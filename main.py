from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import chat
from routes.connectors import gmail

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(chat.router,prefix='/alexa')
app.include_router(gmail.router,prefix='/api/connectors')


@app.get("/cronjob")
def cronjob():
    return {"message":"Server is healthy"}
