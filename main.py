import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import chat
from routes.connectors import gmail
from services.embeddings import warm_up

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # load the embedding model in the background so startup isn't blocked by the download
    warm_up_task = asyncio.create_task(asyncio.to_thread(warm_up))
    yield
    warm_up_task.cancel()


app = FastAPI(lifespan=lifespan)

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
