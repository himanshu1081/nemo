import base64
import html
import os
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from email.message import EmailMessage
from email.utils import getaddresses

from google.auth.transport.requests import AuthorizedSession, Request
from google.oauth2.credentials import Credentials

from services.crypto import decrypt, encrypt
from services.supabase_client import supabase

GMAIL_API = "https://gmail.googleapis.com/gmail/v1/users/me"
TOKEN_URI = "https://oauth2.googleapis.com/token"
MAX_BODY_CHARS = 2000


class GmailNotConnected(Exception):
    pass


class GmailMissingScope(Exception):
    pass


def _parse_expiry(value):
    if not value:
        return None
    expiry = datetime.fromisoformat(value.replace("Z", "+00:00"))
    # google-auth expects naive UTC datetimes
    if expiry.tzinfo:
        expiry = expiry.astimezone(timezone.utc).replace(tzinfo=None)
    return expiry


def get_credentials(user_id: str) -> Credentials:
    result = (
        supabase
        .table("connector_info")
        .select("*")
        .eq("user_id", user_id)
        .eq("provider", "gmail")
        .maybe_single()
        .execute()
    )
    row = result.data if result else None
    if not row:
        raise GmailNotConnected("Gmail is not connected for this user")

    credentials = Credentials(
        token=decrypt(row["access_token"]),
        refresh_token=decrypt(row["refresh_token"]) if row.get("refresh_token") else None,
        token_uri=TOKEN_URI,
        client_id=os.getenv("GOOGLE_CLIENT_ID"),
        client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
        scopes=row.get("scopes"),
        expiry=_parse_expiry(row.get("expires_at")),
    )

    if not credentials.valid:
        credentials.refresh(Request())
        supabase.table("connector_info").update({
            "access_token": encrypt(credentials.token),
            "expires_at": credentials.expiry.isoformat() if credentials.expiry else None,
        }).eq("user_id", user_id).eq("provider", "gmail").execute()

    return credentials


def _session(user_id: str) -> AuthorizedSession:
    return AuthorizedSession(get_credentials(user_id))


def _fetch_metadata(credentials: Credentials, message_ids: list[str], headers: list[str]) -> list[dict]:
    """Fetch message metadata in parallel, one session per request since requests.Session isn't thread safe."""
    def fetch(message_id):
        response = AuthorizedSession(credentials).get(
            f"{GMAIL_API}/messages/{message_id}",
            params={"format": "metadata", "metadataHeaders": headers},
        )
        response.raise_for_status()
        return response.json()

    if not message_ids:
        return []
    with ThreadPoolExecutor(max_workers=len(message_ids)) as pool:
        return list(pool.map(fetch, message_ids))


def _headers(payload: dict) -> dict:
    wanted = {"from", "subject", "date"}
    return {
        h["name"].lower(): h["value"]
        for h in payload.get("headers", [])
        if h["name"].lower() in wanted
    }


def list_messages(user_id: str, query: str = "is:unread", max_results: int = 5) -> list[dict]:
    credentials = get_credentials(user_id)
    response = AuthorizedSession(credentials).get(
        f"{GMAIL_API}/messages",
        params={"q": query, "maxResults": max_results},
    )
    response.raise_for_status()

    ids = [item["id"] for item in response.json().get("messages", [])]
    messages = []
    for data in _fetch_metadata(credentials, ids, ["From", "Subject", "Date"]):
        headers = _headers(data.get("payload", {}))
        messages.append({
            "id": data["id"],
            "from": headers.get("from", ""),
            "subject": headers.get("subject", ""),
            "date": headers.get("date", ""),
            "snippet": html.unescape(data.get("snippet", "")),
        })
    return messages


def _decode(data: str) -> str:
    return base64.urlsafe_b64decode(data + "=" * (-len(data) % 4)).decode("utf-8", errors="ignore")


def _find_part(payload: dict, mime_type: str):
    if payload.get("mimeType") == mime_type and payload.get("body", {}).get("data"):
        return _decode(payload["body"]["data"])
    for part in payload.get("parts", []):
        found = _find_part(part, mime_type)
        if found:
            return found
    return None


def _strip_html(text: str) -> str:
    text = re.sub(r"(?is)<(script|style).*?</\1>", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    return html.unescape(text)


def get_message(user_id: str, message_id: str) -> dict:
    session = _session(user_id)
    response = session.get(f"{GMAIL_API}/messages/{message_id}", params={"format": "full"})
    response.raise_for_status()
    payload = response.json().get("payload", {})

    body = _find_part(payload, "text/plain")
    if not body:
        html_body = _find_part(payload, "text/html")
        body = _strip_html(html_body) if html_body else ""
    body = re.sub(r"\s+", " ", body).strip()[:MAX_BODY_CHARS]

    headers = _headers(payload)
    return {
        "from": headers.get("from", ""),
        "subject": headers.get("subject", ""),
        "date": headers.get("date", ""),
        "body": body,
    }


def find_contacts(user_id: str, name: str, max_results: int = 10) -> list[dict]:
    """Collect email addresses matching a name from the user's sent and received mail."""
    credentials = get_credentials(user_id)
    response = AuthorizedSession(credentials).get(
        f"{GMAIL_API}/messages",
        params={"q": f"from:{name} OR to:{name}", "maxResults": max_results},
    )
    response.raise_for_status()

    ids = [item["id"] for item in response.json().get("messages", [])]
    contacts = {}
    needle = name.lower()
    for data in _fetch_metadata(credentials, ids, ["From", "To", "Cc"]):
        headers = data.get("payload", {}).get("headers", [])
        values = [h["value"] for h in headers if h["name"].lower() in ("from", "to", "cc")]
        for display, address in getaddresses(values):
            if address and (needle in display.lower() or needle in address.lower()):
                contacts.setdefault(address.lower(), {"name": display, "email": address})
    return list(contacts.values())


def send_message(user_id: str, to: str, subject: str, body: str) -> dict:
    message = EmailMessage()
    message["To"] = to
    message["Subject"] = subject
    message.set_content(body)
    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()

    response = _session(user_id).post(f"{GMAIL_API}/messages/send", json={"raw": raw})
    if response.status_code == 403:
        raise GmailMissingScope("Gmail send permission not granted")
    response.raise_for_status()
    return {"id": response.json().get("id")}
