"""Supabase access for conversation memory. Every query is scoped to a user_id.

These calls are synchronous (supabase-py); callers on the event loop wrap them in asyncio.to_thread.
"""
from datetime import datetime, timezone

from services.supabase_client import supabase


class MemoryVersionConflict(Exception):
    pass


def _vector(values: list[float]) -> str:
    # pgvector text format, PostgREST passes it straight to the vector type
    return "[" + ",".join(f"{v:.6f}" for v in values) + "]"


def start_turn(alexa_user_id: str, session_id: str) -> dict:
    """Upsert the user and conversation and return {user_id, conversation_id, memory, memory_version}."""
    result = supabase.rpc("memory_start_turn", {
        "p_alexa_user_id": alexa_user_id,
        "p_session_id": session_id,
    }).execute()
    return result.data[0]


def load_active_messages(user_id: str, conversation_id: str, limit: int) -> list[dict]:
    result = (
        supabase.table("messages")
        .select("id, role, content, metadata")
        .eq("user_id", user_id)
        .eq("conversation_id", conversation_id)
        .is_("summarized_at", "null")
        .order("id", desc=True)
        .limit(limit)
        .execute()
    )
    return list(reversed(result.data))


def load_conversation_summaries(user_id: str, conversation_id: str, limit: int) -> list[str]:
    """Summaries of earlier chunks of this conversation, oldest first."""
    result = (
        supabase.table("conversation_chunks")
        .select("summary")
        .eq("user_id", user_id)
        .eq("conversation_id", conversation_id)
        .eq("status", "done")
        .order("seq", desc=True)
        .limit(limit)
        .execute()
    )
    return [row["summary"] for row in reversed(result.data)]


def match_chunks(user_id: str, embedding: list[float], top_k: int, min_similarity: float,
                 exclude_conversation_id: str | None) -> list[dict]:
    result = supabase.rpc("memory_match_chunks", {
        "p_user_id": user_id,
        "p_query_embedding": _vector(embedding),
        "p_match_count": top_k,
        "p_min_similarity": min_similarity,
        "p_exclude_conversation_id": exclude_conversation_id,
    }).execute()
    return result.data


def save_turn(user_id: str, conversation_id: str, request_id: str | None,
              user_text: str, assistant_text: str, assistant_metadata: dict) -> None:
    # every row needs the same keys: PostgREST fills missing columns in bulk inserts with null
    rows = [
        {"user_id": user_id, "conversation_id": conversation_id, "role": "user",
         "content": user_text, "metadata": {}, "alexa_request_id": request_id},
        {"user_id": user_id, "conversation_id": conversation_id, "role": "assistant",
         "content": assistant_text, "metadata": assistant_metadata, "alexa_request_id": request_id},
    ]
    # a retried Alexa request hits the unique (conversation_id, alexa_request_id, role) and is skipped
    supabase.table("messages").upsert(
        rows,
        on_conflict="conversation_id,alexa_request_id,role",
        ignore_duplicates=True,
    ).execute()


def count_unchunked(user_id: str, conversation_id: str) -> int:
    result = (
        supabase.table("messages")
        .select("id", count="exact", head=True)
        .eq("user_id", user_id)
        .eq("conversation_id", conversation_id)
        .is_("chunk_id", "null")
        .execute()
    )
    return result.count or 0


def conversations_needing_processing(user_id: str, exclude_conversation_id: str | None) -> list[str]:
    """Conversations with messages not yet in a chunk, or with a chunk stuck pending."""
    unchunked = (
        supabase.table("messages")
        .select("conversation_id")
        .eq("user_id", user_id)
        .is_("chunk_id", "null")
    )
    pending = (
        supabase.table("conversation_chunks")
        .select("conversation_id")
        .eq("user_id", user_id)
        .eq("status", "pending")
    )
    if exclude_conversation_id:
        unchunked = unchunked.neq("conversation_id", exclude_conversation_id)
        pending = pending.neq("conversation_id", exclude_conversation_id)
    rows = unchunked.limit(500).execute().data + pending.limit(50).execute().data
    return list(dict.fromkeys(row["conversation_id"] for row in rows))


def claim_chunk(user_id: str, conversation_id: str, chunk_size: int, min_messages: int) -> dict | None:
    result = supabase.rpc("memory_claim_chunk", {
        "p_user_id": user_id,
        "p_conversation_id": conversation_id,
        "p_chunk_size": chunk_size,
        "p_min_messages": min_messages,
    }).execute()
    return result.data[0] if result.data else None


def load_chunk_messages(user_id: str, chunk_id: str) -> list[dict]:
    result = (
        supabase.table("messages")
        .select("role, content")
        .eq("user_id", user_id)
        .eq("chunk_id", chunk_id)
        .order("id")
        .execute()
    )
    return result.data


def load_memory(user_id: str) -> tuple[dict, int]:
    result = (
        supabase.table("user_memory")
        .select("memory, version")
        .eq("user_id", user_id)
        .single()
        .execute()
    )
    return result.data["memory"], result.data["version"]


def complete_chunk(user_id: str, chunk_id: str, summary: str, embedding: list[float],
                   embedding_model: str, memory: dict | None, memory_version: int) -> bool:
    """Returns False if the chunk was already completed (duplicate processing)."""
    try:
        result = supabase.rpc("memory_complete_chunk", {
            "p_user_id": user_id,
            "p_chunk_id": chunk_id,
            "p_summary": summary,
            "p_embedding": _vector(embedding),
            "p_embedding_model": embedding_model,
            "p_memory": memory,
            "p_memory_version": memory_version,
        }).execute()
    except Exception as e:
        if "memory_version_conflict" in str(e):
            raise MemoryVersionConflict() from e
        raise
    return bool(result.data)


def mark_conversation_ended(user_id: str, conversation_id: str) -> None:
    (
        supabase.table("conversations")
        .update({"ended_at": datetime.now(timezone.utc).isoformat()})
        .eq("user_id", user_id)
        .eq("id", conversation_id)
        .execute()
    )


def delete_user(alexa_user_id: str) -> int:
    """Delete everything stored for an Alexa user; foreign keys cascade to all memory tables."""
    result = (
        supabase.table("alexa_users")
        .delete()
        .eq("alexa_user_id", alexa_user_id)
        .execute()
    )
    return len(result.data)


def find_conversation(alexa_user_id: str, session_id: str) -> dict | None:
    """Existing {user_id, id} for an Alexa session, without creating anything."""
    result = (
        supabase.table("conversations")
        .select("id, user_id, alexa_users!inner(alexa_user_id)")
        .eq("alexa_users.alexa_user_id", alexa_user_id)
        .eq("alexa_session_id", session_id)
        .limit(1)
        .execute()
    )
    return result.data[0] if result.data else None
