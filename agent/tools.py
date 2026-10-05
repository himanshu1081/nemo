from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from services import gmail
from services.gmail import GmailNotConnected

NOT_CONNECTED = "Gmail is not connected. Ask the user to connect Gmail in the Nemo web app."


def _user_id(config: RunnableConfig) -> str:
    return config["configurable"]["user_id"]


@tool
def check_emails(config: RunnableConfig, query: str = "is:unread", max_results: int = 5) -> list[dict] | str:
    """List the user's emails matching a Gmail search query.

    Returns id, sender, subject, date and a short snippet for each email.
    Use Gmail search syntax for query, e.g. "is:unread", "from:amazon",
    "subject:invoice", "newer_than:1d", or combine them "is:unread from:github".
    Use an empty query for the latest emails regardless of read state.
    """
    try:
        return gmail.list_messages(_user_id(config), query, min(max_results, 10))
    except GmailNotConnected:
        return NOT_CONNECTED


@tool
def read_email(config: RunnableConfig, message_id: str) -> dict | str:
    """Read the full text of one email, by the id returned from check_emails."""
    try:
        return gmail.get_message(_user_id(config), message_id)
    except GmailNotConnected:
        return NOT_CONNECTED


TOOLS = [check_emails, read_email]
