"""Conversation memory: active messages, chunk summaries with RAG, and long-term user memory.

Per turn:   load_turn_context()  → prompt context, then save_turn() after the reply
Background: process_conversation() turns every 20 messages into a summary + embedding
            and merges durable facts into user_memory
"""
import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from services import memory_store as store
from services.embeddings import embedder
from services.llm import FALLBACK_ERRORS, llm, summary_llm

logger = logging.getLogger(__name__)

CHUNK_SIZE = int(os.getenv("MEMORY_CHUNK_SIZE", "20"))
# active messages can briefly exceed CHUNK_SIZE while a chunk is being summarized
ACTIVE_CONTEXT_LIMIT = int(os.getenv("MEMORY_ACTIVE_CONTEXT_LIMIT", "30"))
# a finished session is summarized even when shorter than a full chunk
SESSION_MIN_MESSAGES = int(os.getenv("MEMORY_SESSION_MIN_MESSAGES", "2"))
RAG_TOP_K = int(os.getenv("MEMORY_RAG_TOP_K", "3"))
RAG_MIN_SIMILARITY = float(os.getenv("MEMORY_RAG_MIN_SIMILARITY", "0.70"))
RAG_MIN_QUERY_WORDS = 3
CURRENT_CONVERSATION_SUMMARIES = 2
TOOL_CONTEXT_TURNS = 3
TOOL_CONTEXT_CHARS = 1500
MAX_FACTS = 40
MAX_PREFERENCES = 15
MAX_CHUNKS_PER_RUN = 5
MAX_MEMORY_CONFLICT_RETRIES = 3


# ---------------------------------------------------------------------------
# Long-term memory shape and updates
# ---------------------------------------------------------------------------

def empty_memory() -> dict:
    return {"name": None, "preferences": {}, "facts": []}


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


class Preference(BaseModel):
    key: str = Field(description="snake_case preference name, e.g. response_style")
    value: str


class FactUpdate(BaseModel):
    index: int = Field(description="number of the existing fact being replaced")
    text: str = Field(description="the corrected fact")


class ChunkAnalysis(BaseModel):
    summary: str = Field(description="information-preserving summary of the conversation chunk")
    name: str | None = Field(description="the user's name if they stated it, otherwise null")
    preferences: list[Preference] = Field(description="preferences the user stated or changed")
    add_facts: list[str] = Field(description="new durable facts about the user")
    update_facts: list[FactUpdate] = Field(description="existing facts that are now outdated, with the replacement")
    remove_facts: list[int] = Field(description="numbers of existing facts that are no longer true")


@dataclass
class MemoryChanges:
    """ChunkAnalysis with fact numbers resolved to text, so it can be re-applied to a newer memory version."""
    name: str | None
    preferences: dict[str, str]
    add: list[str]
    update: list[tuple[str, str]]
    remove: list[str]

    @property
    def empty(self) -> bool:
        return not (self.name or self.preferences or self.add or self.update or self.remove)


def resolve_changes(analysis: ChunkAnalysis, facts: list[str]) -> MemoryChanges:
    def fact_at(index: int) -> str | None:
        return facts[index - 1] if 1 <= index <= len(facts) else None

    return MemoryChanges(
        name=(analysis.name or "").strip() or None,
        preferences={p.key.strip(): p.value.strip() for p in analysis.preferences if p.key.strip() and p.value.strip()},
        add=[f.strip() for f in analysis.add_facts if f.strip()],
        update=[(fact_at(u.index), u.text.strip()) for u in analysis.update_facts if fact_at(u.index) and u.text.strip()],
        remove=[fact_at(i) for i in analysis.remove_facts if fact_at(i)],
    )


def apply_changes(memory: dict, changes: MemoryChanges) -> dict:
    memory = {**empty_memory(), **(memory or {})}
    facts = list(memory["facts"])

    if changes.name:
        memory["name"] = changes.name

    preferences = {**memory["preferences"], **changes.preferences}
    memory["preferences"] = dict(list(preferences.items())[-MAX_PREFERENCES:])

    replacements = {_normalize(old): new for old, new in changes.update}
    removals = {_normalize(old) for old in changes.remove}
    updated = []
    for fact in facts:
        key = _normalize(fact)
        if key in removals:
            continue
        updated.append(replacements.get(key, fact))

    seen = set()
    deduped = []
    for fact in updated + changes.add:
        key = _normalize(fact)
        if key and key not in seen:
            seen.add(key)
            deduped.append(fact)

    # newest facts win when the memory is full
    memory["facts"] = deduped[-MAX_FACTS:]
    return memory


def render_memory(memory: dict) -> str:
    lines = []
    if memory.get("name"):
        lines.append(f"Name: {memory['name']}")
    if memory.get("preferences"):
        lines.append("Preferences: " + "; ".join(f"{k}: {v}" for k, v in memory["preferences"].items()))
    if memory.get("facts"):
        lines.append("Facts:\n" + "\n".join(f"- {fact}" for fact in memory["facts"]))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Summarization pipeline
# ---------------------------------------------------------------------------

SUMMARY_PROMPT = """You maintain the memory of Nemo, a voice assistant on Alexa.

You get the user's current long-term memory and a chunk of conversation. Return:

summary: 2-5 sentences that preserve the useful information in the chunk: topics, decisions,
  requests and their outcomes (e.g. emails sent and to whom), names, dates, numbers and open
  follow-ups. Write in third person ("User asked..."). No filler.

Long-term memory updates. Only durable facts that will still matter in future conversations:
  the user's name, people and contacts they mention, devices and things they own, their work and
  projects, ongoing plans, and stated preferences (e.g. response_style: short).
  Never store one-off requests, small talk, email contents, or anything the assistant said
  that the user didn't confirm. Never invent or infer facts that weren't stated.
- name: only if the user stated their name, else null
- preferences: only preferences the user stated or changed
- add_facts: new facts not already in memory, as short sentences starting with "User"
- update_facts: when the chunk contradicts or refines an existing fact, give its number and the
  corrected text (e.g. "User owns a Philips smart light." replacing the Havells one)
- remove_facts: numbers of existing facts the user said are no longer true
Return empty lists when nothing changed. Most chunks change little or nothing."""


def _analyze_chunk(messages: list[dict], memory: dict) -> ChunkAnalysis:
    facts = memory.get("facts", [])
    numbered = "\n".join(f"{i}. {fact}" for i, fact in enumerate(facts, 1)) or "(none)"
    preferences = json.dumps(memory.get("preferences", {}))
    transcript = "\n".join(f"{m['role']}: {m['content']}" for m in messages)

    analyzer = summary_llm.with_structured_output(ChunkAnalysis, method="json_schema", strict=True).with_fallbacks(
        [llm.with_structured_output(ChunkAnalysis, method="json_schema", strict=True)],
        exceptions_to_handle=FALLBACK_ERRORS,
    )
    return analyzer.invoke([
        SystemMessage(content=SUMMARY_PROMPT),
        HumanMessage(content=(
            f"Current memory\nName: {memory.get('name')}\nPreferences: {preferences}\nFacts:\n{numbered}\n\n"
            f"Conversation chunk\n{transcript}"
        )),
    ])


def _process_chunk(user_id: str, chunk_id: str) -> None:
    messages = store.load_chunk_messages(user_id, chunk_id)
    if not messages:
        logger.warning("Chunk %s has no messages", chunk_id)
        return

    memory, version = store.load_memory(user_id)
    analysis = _analyze_chunk(messages, memory)
    summary = analysis.summary.strip()
    if not summary:
        raise ValueError("empty summary")

    changes = resolve_changes(analysis, memory.get("facts", []))
    embedding = embedder.embed_document(summary)

    for attempt in range(MAX_MEMORY_CONFLICT_RETRIES):
        new_memory = None if changes.empty else apply_changes(memory, changes)
        try:
            completed = store.complete_chunk(
                user_id, chunk_id, summary, embedding, embedder.model_name, new_memory, version
            )
            if not completed:
                logger.info("Chunk %s was already completed by another worker", chunk_id)
            return
        except store.MemoryVersionConflict:
            # memory changed underneath us; re-apply the same changes to the latest version
            logger.info("Memory conflict on chunk %s, retrying (%s)", chunk_id, attempt + 1)
            memory, version = store.load_memory(user_id)
    raise RuntimeError(f"memory still conflicting after {MAX_MEMORY_CONFLICT_RETRIES} retries")


def process_conversation(user_id: str, conversation_id: str, min_messages: int = CHUNK_SIZE) -> None:
    """Summarize every ready chunk of a conversation. Safe to call repeatedly and concurrently."""
    for _ in range(MAX_CHUNKS_PER_RUN):
        try:
            claim = store.claim_chunk(user_id, conversation_id, CHUNK_SIZE, min_messages)
        except Exception:
            logger.exception("Claiming chunk failed for conversation %s", conversation_id)
            return
        if not claim:
            return
        try:
            _process_chunk(user_id, claim["chunk_id"])
            logger.info("Summarized chunk %s (seq %s) of conversation %s", claim["chunk_id"], claim["seq"], conversation_id)
        except Exception:
            # chunk stays pending and is retried once its claim goes stale
            logger.exception("Processing chunk %s failed", claim["chunk_id"])
            return


def flush_previous_conversations(user_id: str, current_conversation_id: str) -> None:
    """Summarize leftovers of earlier sessions, since session-end events aren't guaranteed."""
    try:
        conversation_ids = store.conversations_needing_processing(user_id, current_conversation_id)
    except Exception:
        logger.exception("Finding unprocessed conversations failed")
        return
    for conversation_id in conversation_ids:
        process_conversation(user_id, conversation_id, SESSION_MIN_MESSAGES)


def end_conversation(user_id: str, conversation_id: str) -> None:
    try:
        store.mark_conversation_ended(user_id, conversation_id)
    except Exception:
        logger.exception("Marking conversation %s ended failed", conversation_id)
    process_conversation(user_id, conversation_id, SESSION_MIN_MESSAGES)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve_relevant_memories(user_id: str, query: str, top_k: int = RAG_TOP_K,
                               exclude_conversation_id: str | None = None) -> list[dict]:
    """Most relevant summaries of past conversations for this user, above the similarity threshold."""
    if len(query.split()) < RAG_MIN_QUERY_WORDS:
        return []
    embedding = embedder.embed_query(query)
    matches = store.match_chunks(user_id, embedding, top_k, RAG_MIN_SIMILARITY, exclude_conversation_id)

    seen = set()
    unique = []
    for match in matches:
        key = _normalize(match["summary"])
        if key not in seen:
            seen.add(key)
            unique.append(match)
    return unique


# ---------------------------------------------------------------------------
# Per-turn context
# ---------------------------------------------------------------------------

@dataclass
class TurnContext:
    user_id: str | None = None
    conversation_id: str | None = None
    memory: dict = field(default_factory=empty_memory)
    history: list[dict] = field(default_factory=list)
    conversation_summaries: list[str] = field(default_factory=list)
    relevant_memories: list[dict] = field(default_factory=list)

    @property
    def available(self) -> bool:
        return self.user_id is not None

    @property
    def is_first_turn(self) -> bool:
        return self.available and not self.history and not self.conversation_summaries


async def load_turn_context(alexa_user_id: str, session_id: str, query: str, use_rag: bool = True) -> TurnContext:
    """Load everything the prompt needs. Memory failures degrade to an empty context, never a failed turn."""
    try:
        started = await asyncio.to_thread(store.start_turn, alexa_user_id, session_id)
    except Exception:
        logger.exception("Loading memory failed, answering without it")
        return TurnContext()

    context = TurnContext(
        user_id=started["user_id"],
        conversation_id=started["conversation_id"],
        memory={**empty_memory(), **(started["memory"] or {})},
    )

    async def safe(fn, *args, default):
        try:
            return await asyncio.to_thread(fn, *args)
        except Exception:
            logger.exception("%s failed", fn.__name__)
            return default

    context.history, context.conversation_summaries, context.relevant_memories = await asyncio.gather(
        safe(store.load_active_messages, context.user_id, context.conversation_id, ACTIVE_CONTEXT_LIMIT, default=[]),
        safe(store.load_conversation_summaries, context.user_id, context.conversation_id,
             CURRENT_CONVERSATION_SUMMARIES, default=[]),
        safe(retrieve_relevant_memories, context.user_id, query, RAG_TOP_K, context.conversation_id, default=[])
        if use_rag else asyncio.sleep(0, result=[]),
    )
    return context


def build_context_prompt(context: TurnContext) -> str:
    sections = []
    memory_text = render_memory(context.memory)
    if memory_text:
        sections.append("What you know about the user:\n" + memory_text)
    if context.relevant_memories:
        sections.append(
            "Relevant past conversations (use only if they help answer):\n"
            + "\n".join(f"- {m['summary']}" for m in context.relevant_memories)
        )
    if context.conversation_summaries:
        sections.append("Earlier in this conversation:\n" + "\n".join(f"- {s}" for s in context.conversation_summaries))
    return "\n\n".join(sections)


def build_history(context: TurnContext) -> list[BaseMessage]:
    assistant_indexes = [i for i, m in enumerate(context.history) if m["role"] == "assistant"]
    with_tool_context = set(assistant_indexes[-TOOL_CONTEXT_TURNS:])

    messages = []
    for i, row in enumerate(context.history):
        if row["role"] == "user":
            messages.append(HumanMessage(content=row["content"]))
            continue
        content = row["content"]
        tool_results = (row.get("metadata") or {}).get("tool_results")
        if i in with_tool_context and tool_results:
            content += "\n[Tool data from this turn: " + json.dumps(tool_results)[:TOOL_CONTEXT_CHARS] + "]"
        messages.append(AIMessage(content=content))
    return messages


async def save_turn(context: TurnContext, request_id: str | None, user_text: str,
                    reply: str, tool_results: list[dict]) -> None:
    if not context.available:
        return
    try:
        await asyncio.to_thread(
            store.save_turn, context.user_id, context.conversation_id, request_id,
            user_text, reply, {"tool_results": tool_results} if tool_results else {},
        )
    except Exception:
        logger.exception("Saving turn failed for conversation %s", context.conversation_id)


def needs_chunking(context: TurnContext) -> bool:
    # +2 for the turn just saved; the claim function makes the exact decision
    return context.available and len(context.history) + 2 >= CHUNK_SIZE
