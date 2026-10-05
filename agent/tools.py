import json

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.types import interrupt

from services import gmail
from services.gmail import GmailMissingScope, GmailNotConnected

NOT_CONNECTED = "Gmail is not connected. Ask the user to connect Gmail in the Nemo web app."
MISSING_SCOPE = "Nemo doesn't have permission to send mail. Ask the user to reconnect Gmail in the Nemo web app."
CONFIRM_WORDS = {"yes", "yeah", "yep", "sure", "ok", "okay", "send", "send it", "go ahead", "do it", "confirm"}


def _user_id(config: RunnableConfig) -> str:
    return config["configurable"]["user_id"]


@tool
def check_emails(config: RunnableConfig, query: str = "is:unread", max_results: int = 5) -> str:
    """List the user's emails matching a Gmail search query.

    Returns id, sender, subject, date and a short snippet for each email.
    Use Gmail search syntax for query, e.g. "is:unread", "from:amazon",
    "subject:invoice", "newer_than:1d", or combine them "is:unread from:github".
    Use an empty query for the latest emails regardless of read state.
    """
    try:
        messages = gmail.list_messages(_user_id(config), query, min(max_results, 10))
        return json.dumps(messages) if messages else "No emails found."
    except GmailNotConnected:
        return NOT_CONNECTED


@tool
def read_email(config: RunnableConfig, message_id: str) -> str:
    """Read the full text of one email, by the id returned from check_emails."""
    try:
        return json.dumps(gmail.get_message(_user_id(config), message_id))
    except GmailNotConnected:
        return NOT_CONNECTED


@tool
def find_contact(config: RunnableConfig, name: str) -> str:
    """Find email addresses for a person by name, from the user's past emails.

    Use this before send_email whenever the user names a person instead of
    giving a full email address. If several addresses match, ask the user which one.
    """
    try:
        contacts = gmail.find_contacts(_user_id(config), name)
        return json.dumps(contacts) if contacts else f"No contact named {name} found in past emails. Ask the user for the email address."
    except GmailNotConnected:
        return NOT_CONNECTED


@tool
def send_email(config: RunnableConfig, to: str, subject: str, body: str) -> str:
    """Send an email from the user's Gmail.

    to must be a full email address (use find_contact to look one up).
    Write a short, clear subject and body from what the user asked to say.
    The user is asked to confirm before it is sent; do not ask for confirmation yourself.
    """
    reply = interrupt({"to": to, "subject": subject, "body": body})

    if reply.strip().lower().rstrip(".!") not in CONFIRM_WORDS:
        return f"Not sent. The user replied: \"{reply}\". Change the email if they asked for changes, otherwise acknowledge it was cancelled."

    try:
        gmail.send_message(_user_id(config), to, subject, body)
        return f"Email sent to {to}."
    except GmailNotConnected:
        return NOT_CONNECTED
    except GmailMissingScope:
        return MISSING_SCOPE


TOOLS = [check_emails, read_email, find_contact, send_email]
