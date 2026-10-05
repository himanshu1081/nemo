import os
from fastapi import APIRouter, Request

from agent.agent import run_agent

router = APIRouter()


def speak(text: str, end_session: bool = False) -> dict:
    return {
        "version": "1.0",
        "response": {
            "outputSpeech": {
                "type": "PlainText",
                "text": text
            },
            "shouldEndSession": end_session
        }
    }


def resolve_user_id(body: dict) -> str:
    # Single user for now; swap for Alexa Account Linking later
    return os.getenv("NEMO_USER_ID")


@router.post("")
async def alexa(request: Request):

    print("Server hit on alexa skill")

    body = await request.json()

    request_type = body["request"]["type"]

    if request_type == "LaunchRequest":
        return speak("Hello! I'm Nemo. What would you like to know?")

    if request_type == "SessionEndedRequest":
        return {"version": "1.0", "response": {}}

    if request_type != "IntentRequest":
        return speak("Sorry, I didn't get that.")

    intent = body["request"]["intent"]
    intent_name = intent["name"]

    if intent_name in ("AMAZON.StopIntent", "AMAZON.CancelIntent"):
        return speak("Goodbye!", end_session=True)

    if intent_name == "AMAZON.HelpIntent":
        return speak("You can ask me things like, check my unread emails, or send an email to Rahul.")

    if intent_name == "AMAZON.YesIntent":
        query = "yes"
    elif intent_name == "AMAZON.NoIntent":
        query = "no"
    elif intent_name == "ChatIntent":
        query = intent.get("slots", {}).get("message", {}).get("value")
    else:
        return speak("Sorry, I didn't get that. What would you like to know?")

    if not query:
        return speak("Sorry, I didn't catch that. Could you say it again?")

    session_id = body.get("session", {}).get("sessionId", "default")

    try:
        reply = await run_agent(query, resolve_user_id(body), session_id)
    except Exception as e:
        print("Agent error:", e)
        reply = "Sorry, something went wrong."

    print("Query:", query, "| Reply:", reply)

    return speak(reply)
