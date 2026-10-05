"""Verify that a request really comes from Alexa.

Follows https://developer.amazon.com/en-US/docs/alexa/custom-skills/host-a-custom-skill-as-a-web-service.html
"""
import base64
import json
import os
import posixpath
import warnings
from datetime import datetime, timezone
from urllib.parse import urlparse

import certifi
import httpx
from cryptography import x509
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.x509.verification import PolicyBuilder, Store
from dotenv import load_dotenv
from fastapi import HTTPException, Request

load_dotenv()

ALEXA_SKILL_ID = os.getenv("ALEXA_SKILL_ID")
# Only for local curl testing, never enable in production
SKIP_VERIFICATION = os.getenv("ALEXA_SKIP_VERIFICATION", "").lower() == "true"

CERT_HOST = "s3.amazonaws.com"
CERT_PATH_PREFIX = "/echo.api/"
CERT_SAN = "echo-api.amazon.com"
MAX_REQUEST_AGE_SECONDS = 150

with open(certifi.where(), "rb") as f, warnings.catch_warnings():
    # certifi ships one root with a non-RFC serial number, harmless here
    warnings.simplefilter("ignore")
    _trust_store = Store(x509.load_pem_x509_certificates(f.read()))

_cert_cache: dict[str, list[x509.Certificate]] = {}


def _reject(reason: str):
    print("Rejected Alexa request:", reason)
    raise HTTPException(400, "Invalid Alexa request")


def _check_cert_url(url: str):
    parsed = urlparse(url)
    if parsed.scheme.lower() != "https":
        _reject("cert url scheme")
    if (parsed.hostname or "").lower() != CERT_HOST:
        _reject("cert url host")
    if parsed.port not in (None, 443):
        _reject("cert url port")
    if not posixpath.normpath(parsed.path).startswith(CERT_PATH_PREFIX):
        _reject("cert url path")


async def _get_cert_chain(url: str) -> list[x509.Certificate]:
    chain = _cert_cache.get(url)
    if chain and datetime.now(timezone.utc) < chain[0].not_valid_after_utc:
        return chain

    async with httpx.AsyncClient(timeout=5) as client:
        response = await client.get(url)
    response.raise_for_status()
    chain = x509.load_pem_x509_certificates(response.content)

    # Leaf must chain to a trusted root, be in its validity window and be issued to echo-api.amazon.com
    verifier = (
        PolicyBuilder()
        .store(_trust_store)
        .time(datetime.now(timezone.utc))
        .build_server_verifier(x509.DNSName(CERT_SAN))
    )
    try:
        verifier.verify(chain[0], chain[1:])
    except Exception as e:
        _reject(f"cert chain: {e}")

    _cert_cache[url] = chain
    return chain


async def verify_alexa_request(request: Request) -> dict:
    """Return the parsed request body, or raise 400 if it isn't a genuine Alexa request."""
    raw_body = await request.body()
    try:
        body = json.loads(raw_body)
    except ValueError:
        _reject("body is not json")

    if SKIP_VERIFICATION:
        return body

    application_id = body.get("context", {}).get("System", {}).get("application", {}).get("applicationId")
    if not ALEXA_SKILL_ID or application_id != ALEXA_SKILL_ID:
        _reject("skill id")

    cert_url = request.headers.get("SignatureCertChainUrl")
    signature = request.headers.get("Signature-256")
    if not cert_url or not signature:
        _reject("missing signature headers")

    _check_cert_url(cert_url)
    chain = await _get_cert_chain(cert_url)

    try:
        chain[0].public_key().verify(
            base64.b64decode(signature),
            raw_body,
            padding.PKCS1v15(),
            hashes.SHA256(),
        )
    except Exception:
        _reject("signature")

    try:
        timestamp = datetime.fromisoformat(body["request"]["timestamp"].replace("Z", "+00:00"))
    except (KeyError, ValueError):
        _reject("missing timestamp")
    if abs((datetime.now(timezone.utc) - timestamp).total_seconds()) > MAX_REQUEST_AGE_SECONDS:
        _reject("stale timestamp")

    return body
