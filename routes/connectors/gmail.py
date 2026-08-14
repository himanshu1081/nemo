from fastapi import APIRouter, Request, HTTPException, Query
import os
from supabase import create_client
from google_auth_oauthlib.flow import Flow
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode
import secrets

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_ROLE_KEY")
)

router = APIRouter()

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI")

def create_google_oauth_url(state):
    params = {
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": GOOGLE_REDIRECT_URI,
        "response_type": "code",
        "scope": "https://www.googleapis.com/auth/gmail.readonly",
        "access_type": "offline",
        "state": state,
    }

    return (
        "https://accounts.google.com/o/oauth2/v2/auth?"
        + urlencode(params)
    )

@router.post("/gmail/connect")
async def connectGmail(request:Request):
    authorization = request.headers.get("Authorization")

    if not authorization:
        raise HTTPException(401, "Missing authorization header")

    if not authorization.startswith("Bearer "):
        raise HTTPException(401, "Invalid authorization header")

    access_token = authorization.split(" ",1)[-1]
    user = supabase.auth.get_user(access_token)
    if not user.user:
        raise HTTPException(401, "Invalid Supabase token")
    user_id = user.user.id
    state = secrets.token_urlsafe(32)
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=10)
    supabase.table("connector_info").upsert(
        {
            "user_id": user_id,
            "provider": "gmail",
            "access_token": access_token,
            "refresh_token": refresh_token,
            "expires_at": expires_at,
            "scopes": credentials.scopes
        },
        on_conflict="user_id,provider"
    ).execute()
    google_url = create_google_oauth_url(state)

    return {
        "authorization_url": google_url
    }

@router.get("/gmail/callback")
async def gmail_callback(code: str = Query(...),state: str = Query(...)):
    result = (
        supabase
        .table("oauth_states")
        .select("*")
        .eq("state", state)
        .eq("provider", "gmail")
        .single()
        .execute()
    )
    oauth_state = result.data

    if not oauth_state:
        raise HTTPException(400, "Invalid OAuth state")

    expires_at = datetime.fromisoformat(
        oauth_state["expires_at"].replace("Z", "+00:00")
    )

    if datetime.now(timezone.utc) > expires_at:
        raise HTTPException(400, "OAuth state expired")
    
    user_id = oauth_state["user_id"]

    flow = Flow.from_client_config(
        {
            "web": {
                "client_id": GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
            }
        },
        scopes=[
            "https://www.googleapis.com/auth/gmail.readonly"
        ]
    )

    flow.redirect_uri = GOOGLE_REDIRECT_URI

    flow.fetch_token(code=code)

    credentials = flow.credentials

    access_token = credentials.token
    refresh_token = credentials.refresh_token
    
    #storing both tokens inside supabase
    supabase.table("connector_info").upsert({
        "user_id": user_id,
        "provider": "gmail",
        "access_token": access_token,
        "refresh_token": refresh_token,
        "expires_at": credentials.expiry.isoformat() if credentials.expiry else None,
        "scopes": credentials.scopes
    }).execute()

    #deleting state from supabase
    supabase.table("oauth_states") \
        .delete() \
        .eq("state", state) \
        .execute()

    return {
        "message": "Gmail connected successfully"
    }