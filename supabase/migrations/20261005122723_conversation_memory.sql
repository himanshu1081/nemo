-- Conversation memory for the Nemo Alexa skill.
--
--   alexa_users          one row per Alexa user (Alexa userId), root of all memory
--   conversations        one row per Alexa session
--   messages             raw user/assistant messages (source of truth)
--   conversation_chunks  summaries of 20-message chunks + pgvector embedding for RAG
--   user_memory          compact structured long-term facts/preferences
--
-- Deleting an alexa_users row cascades to everything else (skill disabled → full wipe).
-- Child tables carry user_id and use composite foreign keys so a row can never point at
-- another user's conversation or chunk.
-- RLS is enabled with no policies: only the backend (service role) can touch these tables.

create extension if not exists vector with schema extensions;

-- ---------------------------------------------------------------------------
-- Tables
-- ---------------------------------------------------------------------------

create table public.alexa_users (
    id uuid primary key default gen_random_uuid(),
    alexa_user_id text not null unique,
    -- Supabase auth user (web app account), set once Account Linking exists
    auth_user_id uuid references auth.users (id) on delete set null,
    created_at timestamptz not null default now(),
    last_seen_at timestamptz not null default now()
);

create table public.conversations (
    id uuid primary key default gen_random_uuid(),
    user_id uuid not null references public.alexa_users (id) on delete cascade,
    alexa_session_id text not null,
    started_at timestamptz not null default now(),
    last_message_at timestamptz not null default now(),
    ended_at timestamptz,
    unique (user_id, alexa_session_id),
    unique (id, user_id)
);

create index conversations_user_recent_idx on public.conversations (user_id, last_message_at desc);

create table public.conversation_chunks (
    id uuid primary key default gen_random_uuid(),
    user_id uuid not null,
    conversation_id uuid not null,
    seq integer not null,
    status text not null default 'pending' check (status in ('pending', 'done', 'failed')),
    summary text,
    embedding extensions.vector(384),
    embedding_model text,
    message_count integer not null,
    first_message_at timestamptz not null,
    last_message_at timestamptz not null,
    attempts integer not null default 0,
    claimed_at timestamptz not null default now(),
    created_at timestamptz not null default now(),
    completed_at timestamptz,
    unique (conversation_id, seq),
    unique (id, user_id),
    foreign key (conversation_id, user_id)
        references public.conversations (id, user_id) on delete cascade,
    check (status <> 'done' or (summary is not null and embedding is not null))
);

create index conversation_chunks_user_idx
    on public.conversation_chunks (user_id, created_at desc) where status = 'done';
create index conversation_chunks_pending_idx
    on public.conversation_chunks (conversation_id) where status = 'pending';
-- ANN index for larger users; for small per-user sets the planner uses the user_id index
create index conversation_chunks_embedding_idx
    on public.conversation_chunks using hnsw (embedding extensions.vector_cosine_ops)
    where status = 'done';

create table public.messages (
    id bigint generated always as identity primary key,
    user_id uuid not null,
    conversation_id uuid not null,
    role text not null check (role in ('user', 'assistant')),
    content text not null,
    -- assistant turns keep compact tool results so follow-ups ("read the first one") work
    metadata jsonb not null default '{}'::jsonb,
    alexa_request_id text,
    chunk_id uuid,
    summarized_at timestamptz,
    created_at timestamptz not null default now(),
    foreign key (conversation_id, user_id)
        references public.conversations (id, user_id) on delete cascade,
    foreign key (chunk_id, user_id)
        references public.conversation_chunks (id, user_id) on delete set null (chunk_id),
    -- Alexa request retries must not duplicate a turn
    unique (conversation_id, alexa_request_id, role)
);

create index messages_active_idx
    on public.messages (conversation_id, id) where summarized_at is null;
create index messages_unchunked_idx
    on public.messages (user_id, conversation_id) where chunk_id is null;
create index messages_chunk_idx on public.messages (chunk_id);

create table public.user_memory (
    user_id uuid primary key references public.alexa_users (id) on delete cascade,
    memory jsonb not null default '{"name": null, "preferences": {}, "facts": []}'::jsonb,
    -- optimistic concurrency for concurrent chunk processing
    version integer not null default 0,
    updated_at timestamptz not null default now()
);

-- ---------------------------------------------------------------------------
-- Access control: backend only
-- ---------------------------------------------------------------------------

alter table public.alexa_users enable row level security;
alter table public.conversations enable row level security;
alter table public.conversation_chunks enable row level security;
alter table public.messages enable row level security;
alter table public.user_memory enable row level security;

revoke all on public.alexa_users, public.conversations, public.conversation_chunks,
    public.messages, public.user_memory from anon, authenticated;

-- ---------------------------------------------------------------------------
-- Functions
-- ---------------------------------------------------------------------------

-- One round trip at the start of every turn: upsert user + conversation, return memory.
create or replace function public.memory_start_turn(p_alexa_user_id text, p_session_id text)
returns table (user_id uuid, conversation_id uuid, memory jsonb, memory_version integer)
language plpgsql
set search_path = public
as $$
#variable_conflict use_column
declare
    v_user_id uuid;
    v_conversation_id uuid;
begin
    insert into alexa_users (alexa_user_id)
    values (p_alexa_user_id)
    on conflict (alexa_user_id) do update set last_seen_at = now()
    returning id into v_user_id;

    insert into user_memory (user_id) values (v_user_id) on conflict do nothing;

    insert into conversations (user_id, alexa_session_id)
    values (v_user_id, p_session_id)
    on conflict (user_id, alexa_session_id) do update set last_message_at = now()
    returning id into v_conversation_id;

    return query
    select v_user_id, v_conversation_id, m.memory, m.version
    from user_memory m where m.user_id = v_user_id;
end;
$$;

-- Atomically claim the oldest unchunked messages of a conversation as a new chunk.
-- Returns a stale pending chunk first (crashed worker), so retries pick up where they left off.
-- Returns no row when there is nothing to do.
create or replace function public.memory_claim_chunk(
    p_user_id uuid,
    p_conversation_id uuid,
    p_chunk_size integer,
    p_min_messages integer,
    p_stale_after interval default interval '5 minutes',
    p_max_attempts integer default 3
)
returns table (chunk_id uuid, seq integer)
language plpgsql
set search_path = public
as $$
#variable_conflict use_column
declare
    v_chunk_id uuid;
    v_seq integer;
    v_ids bigint[];
begin
    -- serialize chunking per conversation
    perform 1 from conversations c
    where c.id = p_conversation_id and c.user_id = p_user_id
    for update;
    if not found then
        return;
    end if;

    -- retry a stale claim
    update conversation_chunks ch
    set claimed_at = now(), attempts = ch.attempts + 1
    where ch.id = (
        select s.id from conversation_chunks s
        where s.conversation_id = p_conversation_id and s.user_id = p_user_id
          and s.status = 'pending' and s.claimed_at < now() - p_stale_after
        order by s.seq limit 1
    )
    returning ch.id, ch.seq into v_chunk_id, v_seq;

    if v_chunk_id is not null then
        if (select attempts from conversation_chunks where id = v_chunk_id) > p_max_attempts then
            -- stop retrying; archive the raw messages so the active context can't grow forever
            update conversation_chunks set status = 'failed' where id = v_chunk_id;
            update messages set summarized_at = now() where chunk_id = v_chunk_id;
            return;
        end if;
        return query select v_chunk_id, v_seq;
        return;
    end if;

    select array_agg(m.id order by m.id) into v_ids
    from (
        select id from messages
        where conversation_id = p_conversation_id and user_id = p_user_id and chunk_id is null
        order by id
        limit p_chunk_size
    ) m;

    if coalesce(array_length(v_ids, 1), 0) < p_min_messages then
        return;
    end if;

    select coalesce(max(s.seq), 0) + 1 into v_seq
    from conversation_chunks s where s.conversation_id = p_conversation_id;

    insert into conversation_chunks
        (user_id, conversation_id, seq, message_count, first_message_at, last_message_at, attempts)
    select p_user_id, p_conversation_id, v_seq, count(*), min(created_at), max(created_at), 1
    from messages where id = any (v_ids)
    returning id into v_chunk_id;

    update messages set chunk_id = v_chunk_id where id = any (v_ids);

    return query select v_chunk_id, v_seq;
end;
$$;

-- Finish a chunk: store summary + embedding, archive its messages and save the updated
-- memory in one transaction. Idempotent: returns false if the chunk was already completed.
-- Raises 'memory_version_conflict' if memory changed since it was read (caller re-applies).
create or replace function public.memory_complete_chunk(
    p_user_id uuid,
    p_chunk_id uuid,
    p_summary text,
    p_embedding extensions.vector(384),
    p_embedding_model text,
    p_memory jsonb,
    p_memory_version integer
)
returns boolean
language plpgsql
set search_path = public, extensions
as $$
begin
    update conversation_chunks
    set status = 'done', summary = p_summary, embedding = p_embedding,
        embedding_model = p_embedding_model, completed_at = now()
    where id = p_chunk_id and user_id = p_user_id and status = 'pending';
    if not found then
        return false;
    end if;

    update messages set summarized_at = now()
    where chunk_id = p_chunk_id and user_id = p_user_id;

    if p_memory is not null then
        update user_memory
        set memory = p_memory, version = version + 1, updated_at = now()
        where user_id = p_user_id and version = p_memory_version;
        if not found then
            raise exception 'memory_version_conflict';
        end if;
    end if;

    return true;
end;
$$;

-- Semantic search over one user's conversation summaries.
create or replace function public.memory_match_chunks(
    p_user_id uuid,
    p_query_embedding extensions.vector(384),
    p_match_count integer default 3,
    p_min_similarity double precision default 0.7,
    p_exclude_conversation_id uuid default null
)
returns table (
    id uuid,
    conversation_id uuid,
    summary text,
    similarity double precision,
    last_message_at timestamptz
)
language sql
stable
set search_path = public, extensions
as $$
    select ch.id, ch.conversation_id, ch.summary,
           1 - (ch.embedding <=> p_query_embedding) as similarity,
           ch.last_message_at
    from conversation_chunks ch
    where ch.user_id = p_user_id
      and ch.status = 'done'
      and (p_exclude_conversation_id is null or ch.conversation_id <> p_exclude_conversation_id)
      and 1 - (ch.embedding <=> p_query_embedding) >= p_min_similarity
    order by ch.embedding <=> p_query_embedding
    limit least(p_match_count, 10);
$$;

revoke execute on function public.memory_start_turn(text, text) from public, anon, authenticated;
revoke execute on function public.memory_claim_chunk(uuid, uuid, integer, integer, interval, integer) from public, anon, authenticated;
revoke execute on function public.memory_complete_chunk(uuid, uuid, text, extensions.vector, text, jsonb, integer) from public, anon, authenticated;
revoke execute on function public.memory_match_chunks(uuid, extensions.vector, integer, double precision, uuid) from public, anon, authenticated;
grant execute on function public.memory_start_turn(text, text) to service_role;
grant execute on function public.memory_claim_chunk(uuid, uuid, integer, integer, interval, integer) to service_role;
grant execute on function public.memory_complete_chunk(uuid, uuid, text, extensions.vector, text, jsonb, integer) to service_role;
grant execute on function public.memory_match_chunks(uuid, extensions.vector, integer, double precision, uuid) to service_role;
