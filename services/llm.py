import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
    # reasoning tokens count toward max_tokens, keep effort low for Alexa's timeout
    reasoning_effort="low",
    max_tokens=1024
)
