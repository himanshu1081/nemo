import os
from dotenv import load_dotenv
from groq import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError
from langchain_groq import ChatGroq

load_dotenv()

# Errors that make us switch models immediately instead of waiting on retries
FALLBACK_ERRORS = (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError)

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
    # reasoning tokens count toward max_tokens, keep effort low for Alexa's timeout
    reasoning_effort="low",
    max_tokens=1024,
    # fail fast so the fallback still answers inside Alexa's ~8s window
    max_retries=0,
    timeout=5,
)

small_llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
    reasoning_effort="low",
    max_tokens=1024,
    max_retries=1,
    timeout=5,
)

# Background summarization isn't latency bound, so it retries patiently instead of failing fast
summary_llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
    reasoning_effort="low",
    max_tokens=2048,
    max_retries=4,
    timeout=30,
)
