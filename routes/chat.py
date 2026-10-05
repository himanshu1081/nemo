import logging
import os
from fastapi import APIRouter, BackgroundTasks, Request

from agent import memory
from agent.agent import end_session, run_agent
from services import memory_store
from services.alexa_verify import verify_alexa_request

logger = logging.getLogger(__name__)

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


def alexa_user_id(body: dict) -> str | None:
    return body.get("context", {}).get("System", {}).get("user", {}).get("userId")


def delete_user_memory(user: str) -> None:
    try:
        deleted = memory_store.delete_user(user)
        logger.info("Skill disabled, deleted memory for %s user row(s)", deleted)
    except Exception:
        # Alexa doesn't retry skill events, so this needs attention if it ever fails
        logger.exception("Deleting memory after skill disable FAILED")


def finish_session(user: str, session_id: str) -> None:
    """Session-end checkpoint: summarize whatever is left. Not guaranteed to run, so it's only a bonus."""
    try:
        conversation = memory_store.find_conversation(user, session_id)
    except Exception:
        logger.exception("Looking up conversation for session end failed")
        return
    if conversation:
        memory.end_conversation(conversation["user_id"], conversation["id"])


@router.post("")
async def alexa(request: Request, background_tasks: BackgroundTasks):

    print("Server hit on alexa skill")

    body = await verify_alexa_request(request)

    request_type = body["request"]["type"]
    user = alexa_user_id(body)
    session_id = body.get("session", {}).get("sessionId", "default")

    if request_type == "AlexaSkillEvent.SkillDisabled":
        if user:
            background_tasks.add_task(delete_user_memory, user)
        return {"version": "1.0", "response": {}}

    if request_type == "LaunchRequest":
        return speak("Hello! I'm Nemo. What would you like to know?")

    if request_type == "SessionEndedRequest":
        await end_session(session_id)
        if user:
            background_tasks.add_task(finish_session, user, session_id)
        return {"version": "1.0", "response": {}}

    if request_type != "IntentRequest":
        return speak("Sorry, I didn't get that.")

    intent = body["request"]["intent"]
    intent_name = intent["name"]

    if intent_name in ("AMAZON.StopIntent", "AMAZON.CancelIntent"):
        await end_session(session_id)
        if user:
            background_tasks.add_task(finish_session, user, session_id)
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

    if not user:
        return speak("Sorry, I couldn't identify you.")

    try:
        result = await run_agent(query, user, session_id, body["request"].get("requestId"), resolve_user_id(body))
    except Exception:
        logger.exception("Agent error")
        return speak("Sorry, something went wrong.")

    context = result.context
    if context.is_first_turn:
        background_tasks.add_task(memory.flush_previous_conversations, context.user_id, context.conversation_id)
    if memory.needs_chunking(context):
        background_tasks.add_task(memory.process_conversation, context.user_id, context.conversation_id)

    print("Query:", query, "| Reply:", result.text)

    return speak(result.text)
