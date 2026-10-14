# AI-Powered Vault Document System

Upload documents, get AI-generated insights, and chat with them in natural
language. Answers are grounded in the documents and cite the exact page they
came from; when the documents do not contain an answer, the system says so
instead of inventing one.

---

## Quick start

Requirements: Docker Desktop (or Docker Engine + Compose) and an OpenAI API key.

```bash
cp .env.example .env          # then set OPENAI_API_KEY
docker compose up -d --build
```

That starts four services - API, Celery worker, PostgreSQL with pgvector, and
Redis - and applies the database migrations automatically.

| URL | What it is |
|---|---|
| http://localhost:8000/docs | Interactive API documentation (Swagger) |
| http://localhost:8000/ui | Minimal chat interface |
| http://localhost:8000/health | Dependency health check |

Every `/api/v1` request needs an API key header: `X-API-Key: dev-key-change-me`
(configurable through `APP_API_KEYS`).

### Try it end to end

```bash
KEY="X-API-Key: dev-key-change-me"
BASE=http://localhost:8000/api/v1

# 1. Upload a document (returns immediately; processing runs in the background)
curl -X POST $BASE/documents/upload -H "$KEY" -F "file=@your-document.pdf"

# 2. Poll until status is ready
curl $BASE/documents/<document-id> -H "$KEY"

# 3. Read the AI analysis
curl $BASE/documents/<document-id>/insights -H "$KEY"

# 4. Start a chat session (omit document_ids to chat across everything)
curl -X POST $BASE/chat/sessions -H "$KEY" -H "Content-Type: application/json" \
  -d '{"document_ids": ["<document-id>"]}'

# 5. Ask a question
curl -X POST $BASE/chat/sessions/<session-id>/messages \
  -H "$KEY" -H "Content-Type: application/json" \
  -d '{"question": "What is the expense limit?"}'
```

---

## How it works

```
Upload -> API validates, stores the file, returns 202 -> Celery queue
                                                              |
            +-------------------------------------------------+
            v
  extract -> chunk -> embed -> analyze -> status: ready
  (pages)   (800 tok) (vectors) (insights)

Ask -> condense question -> vector search -> relevance gate -> answer + citations
                                                   |
                                                   +- below threshold: say so
```

**Upload is non-blocking.** The API validates the file, writes it to storage and
returns `202 Accepted` with a document id. All the slow work happens in the
Celery worker, so a 200-page PDF never holds an HTTP connection open.

**Each pipeline stage is timed and checkpointed.** Stage results are written to
`processing_metrics`, so a failure is attributable to extraction, chunking,
embedding or analysis rather than to "processing" as a whole. Stages are
individually idempotent, which makes retries safe.

**Chunks keep their page number.** That is what allows an answer to cite
"policy.pdf, page 2" and quote the passage it relied on.

**Retrieval is hybrid, and the split is measured.** Dense vectors and
exact-term search fail in opposite directions: embeddings generalise well but
dilute rare terms inside heterogeneous text such as policy schedules, while
PostgreSQL full-text search nails those terms but misses paraphrase. On the
labelled set in `eval/`, semantic retrieval scores recall@6 of 82 percent
against lexical's 9 percent, and fusing them reaches 91 percent. Lexical is not
carrying the system - it recovers the specific questions embeddings lose.

**Fusion needs deep candidate lists.** Both arms return `RAG_CANDIDATE_POOL`
candidates, not `top_k`. Reciprocal rank fusion rewards chunks both rankers
found, which cannot happen if each list is truncated to the handful that will
be shown: at depth 6 the two lists were entirely disjoint, so fusion degenerated
into concatenation and the lexical arm contributed nothing.

**Fusion weights are equal.** Down-weighting the lexical arm looks reasonable
given its lower standalone recall, but it breaks fusion arithmetic. A chunk only
exact-term search found enters at lexical rank 1, and any weight below
`(RRF_K + 1) / (RRF_K + top_k)` - 0.924 at the default `top_k` - scores it below
the weakest semantic hit, so it can never reach the model. Lexical's lower
precision is handled by the relevance gate instead.

**Retrieval has a relevance gate.** Vector search always returns its nearest
neighbours, even for a question the corpus cannot answer. A question is only
answered if the vector score clears `RAG_MIN_SIMILARITY` *or* the documents
literally contain the terms asked about; otherwise the system says so rather
than reasoning over noise.

**Questions are classified before retrieval.** Questions about the conversation
itself ("what did I ask?") are answered from the transcript instead of being
refused by a gate that was never going to match. Counting and listing questions
retrieve more widely and are required to state that the excerpts are a sample.

### Where the code lives

| Step | Component |
|---|---|
| HTTP routing, validation, auth | `app/api/` |
| Business logic | `app/services/` |
| Database access | `app/repositories/` |
| ORM models | `app/models/` |
| AI providers, prompts, RAG | `app/ai/` |
| Extraction, chunking, pipeline | `app/processing/` |
| Object storage | `app/storage/` |
| Background tasks | `app/workers/` |

Every external dependency sits behind an interface - `StorageBackend`,
`LLMProvider`, `EmbeddingProvider`, `TextExtractor` - so implementations can be
swapped without touching business logic, and tests run without network access.

---

## API

All endpoints are under `/api/v1` and require `X-API-Key`.

### Documents

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/documents/upload` | Upload a PDF, DOCX, TXT or MD file |
| `GET` | `/documents` | List documents (filter by status, category, tag, search) |
| `GET` | `/documents/{id}` | Metadata, processing status, insights |
| `GET` | `/documents/{id}/insights` | Summary, key points, tags, entities, sentiment |
| `POST` | `/documents/{id}/reprocess` | Re-run the pipeline |
| `DELETE` | `/documents/{id}` | Delete the document, its chunks and its file |

### Chat

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/chat/sessions` | Start a session over selected documents, or all of them |
| `GET` | `/chat/sessions` | List sessions |
| `GET` | `/chat/sessions/{id}` | Session detail |
| `POST` | `/chat/sessions/{id}/messages` | Ask a question; returns an answer with citations |
| `POST` | `/chat/sessions/{id}/messages/stream` | Same, streamed as Server-Sent Events |
| `GET` | `/chat/sessions/{id}/messages` | Conversation history |
| `DELETE` | `/chat/sessions/{id}` | Delete a session |

Answers carry a `source` field: `documents` when the answer came from
retrieved excerpts, `conversation` when the question was about the exchange
itself and was answered from the transcript. A refusal is `documents` with
`grounded: false` and no citations.

### Monitoring

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/metrics/documents` | Corpus totals, status/type/category breakdowns, top tags |
| `GET` | `/metrics/processing` | Per-stage run counts, success rates, avg/p95/max duration |
| `GET` | `/metrics/ai-usage` | Token consumption and USD cost by operation and model |
| `GET` | `/health` | Database, Redis, storage and AI provider status (no auth) |

### Errors

Every failure returns the same envelope, so clients parse one shape:

```json
{
  "error": {
    "code": "document_not_ready",
    "message": "These documents are not ready yet: 3fa85f64-...",
    "details": { "document_ids": ["3fa85f64-..."] }
  }
}
```

Codes include `unsupported_file_type`, `file_too_large`, `empty_file`,
`document_not_ready`, `not_found`, `authentication_error`, `rate_limited`,
`llm_rate_limited` and `text_extraction_failed`.

---

## Configuration

Settings are grouped by prefix in `.env`; see `.env.example` for the full list.

| Variable | Default | Notes |
|---|---|---|
| `OPENAI_API_KEY` | - | Required |
| `OPENAI_CHAT_MODEL` | `gpt-4o-mini` | Raise to `gpt-4o` for higher quality |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | 1536 dimensions |
| `APP_API_KEYS` | `dev-key-change-me` | Comma-separated accepted keys |
| `STORAGE_BACKEND` | `local` | `local` or `s3` |
| `INGEST_MAX_UPLOAD_MB` | `25` | Upload size limit |
| `INGEST_CHUNK_TOKENS` | `350` | Chunk size, with 18 percent overlap |
| `RAG_TOP_K` | `6` | Chunks retrieved per question |
| `RAG_AGGREGATE_TOP_K` | `12` | Chunks retrieved for counting/listing questions |
| `RAG_CANDIDATE_POOL` | `40` | Candidates fetched per arm before fusion |
| `RAG_MIN_SIMILARITY` | `0.25` | Vector score below which semantic evidence is rejected |
| `RAG_MIN_LEXICAL_RANK` | `0.02` | Full-text rank below which exact-term evidence is rejected |
| `RATE_LIMIT_REQUESTS` | `120` | Per API key, per minute |
| `RATE_LIMIT_ENABLED` | `true` | Set false to disable limiting |

Changing `.env` requires `docker compose up -d --force-recreate`; a plain
`restart` does not re-read the file.

---

## Storage backends

Document bytes go through a `StorageBackend` interface with two implementations:

- **`local`** (default) - the filesystem, in a Docker volume. No extra service,
  which keeps development and tests fast.
- **`s3`** - AWS S3 via boto3. Set `STORAGE_BACKEND=s3` plus the `S3_*`
  variables. On EC2 point it at a real bucket; locally, LocalStack is available
  as an optional profile:

  ```bash
  docker compose --profile s3 up -d
  ```

The ingestion pipeline is identical either way.

---

## Deployment

The stack is self-contained and runs the same on a laptop and on an EC2
instance:

```bash
# On the instance, with Docker installed
git clone <repo> && cd <repo>
cp .env.example .env            # set OPENAI_API_KEY and APP_API_KEYS
docker compose up -d --build
```

Documents and database contents live in named volumes, so restarts and
redeploys preserve data. Expose only port 8000 through the security group, and
front it with nginx or Caddy for TLS if it is publicly reachable. A `t3.large`
(2 vCPU / 8 GB) comfortably runs the whole stack.

**Scaling path.** Each compose service maps onto a managed AWS equivalent with
configuration changes only: PostgreSQL to RDS (pgvector is supported), Redis to
ElastiCache, the API and worker to ECS Fargate behind an ALB, local storage to
S3, and `.env` to Secrets Manager. Workers scale independently of the API
because they communicate only through the queue.

---

## Testing

```bash
docker compose exec api python -m pytest tests -q          # full suite
docker compose exec api python -m pytest tests -q --cov=app
docker compose exec api python -m eval.run_retrieval_eval  # retrieval quality
```

397 tests, 88% line coverage, running in about 5 seconds.

Tests never call the OpenAI API: providers are replaced with doubles that return
deterministic vectors and canned completions, so the suite is fast, free and
offline.

- **Unit tests** cover extraction, chunking, pipeline stages, retrieval,
  citation mapping, chat orchestration, streaming, caching, rate limiting and
  both storage backends.
### Retrieval evaluation

Unit tests prove retrieval does what the code says. They cannot tell you whether
it finds the right passage, so `eval/` holds a labelled set: plain-language
questions paired by hand with the chunks that answer them, plus control
questions the corpus cannot answer and must refuse.

```
recall@6 semantic   9/11 = 82%
recall@6 lexical    1/11 =  9%
recall@6 hybrid    10/11 = 91%
controls refused   2/2
```

The runner exits non-zero below `MIN_HYBRID_RECALL`, so it can gate a pipeline.
Reporting each arm separately is deliberate: a single blended number hides a
change that helps one arm while silently breaking the other, which is exactly
the regression this harness was written after finding.

Questions are phrased the way a user would ask them rather than in the
document's vocabulary. Questions that echo the source text are trivially
retrievable by either arm and measure nothing.

- **Integration tests** run against the real PostgreSQL service: repository SQL
  (filters, JSONB containment, joins), pgvector cosine similarity, the full
  ingestion pipeline, and the upload path. Each gets its own transaction, which
  is rolled back afterwards, so they need no cleanup and cannot interfere with
  each other.

---

## Design decisions

**Chunks are split per page, not per document.** A chunk spanning two pages
cannot be cited precisely. Splitting per page costs a little chunking
efficiency and buys exact citations.

**A relevance threshold, not just top-k.** Returning the nearest chunks for
every question makes the model answer from irrelevant text. Thresholding turns
no-good-match into an honest refusal.

**Follow-up questions are rewritten before retrieval.** "And what about
contractors?" embeds poorly on its own. A condensing call turns it into a
standalone query, which is what makes multi-turn chat retrieve correctly. The
call is skipped on the first turn, and falls back to the raw question if the
provider fails.

**Citations are mapped back from the answer.** Only excerpts the model actually
cited are returned, so the citation list is evidence for the answer rather than
a dump of everything retrieval happened to find. Markers are then renumbered
sequentially, so an answer citing retrieval slots 4-6 displays [1][2][3] above
three sources rather than referring to a [1] that is not there.

**Citation snippets are centred on the supporting passage.** Quoting the start
of a chunk means the evidence shown often does not contain the claim; the
snippet is instead a window around the sentence that best matches the question.

**Extraction artefacts are repaired before chunking.** Some PDFs simulate bold
by overprinting glyphs, which extracts as "SSSSBBBBIIII" for "SBI". Left alone
it wastes tokens, corrupts citations, and - because the same artefact prefixes
every chunk - makes chunks embed more similarly to one another. Only tokens
whose every character repeats a uniform number of times are collapsed, so
ordinary words and numbers are untouched.

**Analysis runs last in the pipeline.** If it fails, the document is still
chunked, embedded and fully chattable - only the summary is missing.

**Sessions are scoped by an association table, not a JSON array of ids.** That
gives referential integrity and lets retrieval filter by join.

**Structured outputs instead of JSON parsing.** The provider enforces the
schema, so there is no repair-and-retry path for malformed model output.

**Duplicate uploads are detected by content hash.** Re-uploading the same file
returns the existing document instead of paying to process it twice.

**The streaming endpoint opens its own database session.** A streamed response
is generated after the endpoint returns, by which point a request-scoped
session has already closed - so the conversation would never be persisted.

**Query embeddings are cached in Redis.** The same question embeds to the same
vector, so a repeat costs nothing. The key includes the model, so changing the
embedding model cannot serve vectors of the wrong width.

**Enums are stored as VARCHAR with a type decorator.** A native Postgres enum
locks the type and needs a migration to add a value; this keeps the vocabulary
owned by the application while still returning real enum members in Python.

### Assumptions

- Single-tenant: the API key identifies the user. Every table carries `user_id`
  and every query is scoped by it, so multi-tenant auth is a change to one
  module.
- Text-based documents only. Scanned images would need OCR, which is out of
  scope.
- Legacy `.doc` is not supported; `.docx` is.
- Word and plain-text files have no real pages, so they are divided into
  roughly 3000-character synthetic pages to keep citations specific.
- The rate limiter uses a fixed window, so a client straddling a boundary can
  briefly exceed the nominal rate. That is an accepted trade-off for one Redis
  round trip per request.

---

## Project layout

```
app/
  api/            FastAPI routers, middleware, error handlers, rate limiting
  ai/             Provider interfaces, OpenAI implementations, prompts, RAG
    rag/          Retrieval, context assembly, citation mapping
  core/           Configuration, logging, exceptions, security, constants
  db/             Engines, session management, base models, custom types
  models/         SQLAlchemy ORM models
  processing/     Extractors, chunker, ingestion pipeline stages
  repositories/   Data access
  schemas/        Pydantic request and response models
  services/       Business logic
  storage/        StorageBackend interface, local and S3 implementations
  workers/        Celery application and tasks
alembic/          Database migrations
eval/             Labelled retrieval set and recall harness
static/           Minimal chat UI
tests/            Unit and integration tests
```

See `AI_USAGE.md` for how AI tooling was used to build this, and `Docs/` for the
original project plan.

---

## Streaming

For a responsive UI, answers can be streamed as Server-Sent Events:

```bash
curl -N -X POST $BASE/chat/sessions/<session-id>/messages/stream \
  -H "$KEY" -H "Content-Type: application/json" \
  -d '{"question": "What is the expense limit?"}'
```

```
event: token
data: {"text": "You"}

event: token
data: {"text": " can claim"}

event: citations
data: {"citations": [{"marker": 1, "document_name": "policy.pdf", "page_number": 2, ...}]}

event: done
data: {"message_id": "...", "grounded": true, "source": "documents", "latency_ms": 2140, "model": "gpt-4o-mini"}
```

Citations arrive at the end rather than alongside the text, because which
sources an answer relies on is only known once the model has finished citing
them. The web UI uses this endpoint, re-rendering the message with clickable
citation markers when the `done` event arrives.
