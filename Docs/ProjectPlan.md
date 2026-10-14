# AI-Powered Vault Document System — Project Plan

> **Status:** APPROVED (v1.1) — decisions confirmed with stakeholder on 2026-10-13; ready for execution.
> **Source:** `Requirements/AI-Powered Vault Document System_ Case Study.pdf`
> **Date:** 2026-10-13

---

## 1. Executive Summary

Build an **AI-first backend** ("ChatGPT for your documents") where users:

1. **Upload** documents (PDF, DOCX, TXT) → processed **asynchronously** (extract → chunk → embed → analyze).
2. Get **AI insights** per document (summary, key points, tags, entities, sentiment).
3. **Chat** with one or more documents via **RAG** — contextual answers with **source citations**, multi-turn memory, and follow-up suggestions.
4. **Monitor** the system via metrics APIs (document stats, processing performance, AI cost tracking).

This is an evaluation case study scored with **equal weight (33% each)** on:
- **AI-First Development** — AI integration quality, prompt engineering, use of AI coding tools (documented in `AI_USAGE.md`)
- **Product Thinking** — intuitive APIs, prioritization, error handling, real-world usability
- **Technical Implementation** — code quality, async processing, API design, docs

**Guiding principle from the brief:** *"We prefer a fully functional subset over partially implemented features."* → We build a rock-solid core first, then layer bonus features.

---

## 2. Business Analysis

### 2.1 Problem Statement
Knowledge workers accumulate documents whose content is locked away — finding answers means manually reading. The system must turn a passive document store ("vault") into an **interactive knowledge base**: ask a question in natural language, get an accurate, **cited** answer in seconds.

### 2.2 Personas
| Persona | Need | Key APIs |
|---|---|---|
| **End user** (knowledge worker) | Upload docs, ask questions, trust the answers (citations) | Upload, Chat, Insights |
| **Product/Ops** | Know system health, usage, AI spend | Metrics, Health |
| **Developer/Integrator** | Clear, predictable, well-documented API | OpenAPI docs, consistent errors |

### 2.3 Core User Stories (MoSCoW-prioritized)

**Must Have (MVP — the demo depends on these)**
- US-01: As a user, I can upload a PDF/DOCX/TXT and immediately get a document ID + processing status (non-blocking).
- US-02: As a user, I can check a document's processing status (`queued → processing → ready / failed`) and see a failure reason.
- US-03: As a user, I can list my documents and retrieve metadata + AI insights (summary, key points, tags).
- US-04: As a user, I can start a chat session scoped to one or more documents.
- US-05: As a user, I can ask a question and receive an answer **grounded in the document with citations** (doc, page/chunk, quoted snippet).
- US-06: As a user, I can ask follow-up questions that understand prior turns (multi-turn context).
- US-07: As a user, I can retrieve chat history for a session.
- US-08: As an operator, I can call metrics endpoints for document statistics and processing performance.
- US-09: As a user, when I ask something not in the documents, the system says so instead of hallucinating.

**Should Have (high-value bonus, low marginal cost)**
- US-10: Multi-document chat (question across all my documents or a selected set).
- US-11: Follow-up question suggestions after each answer.
- US-12: Automated categorization & tagging on ingest.
- US-13: Streaming chat responses (SSE).
- US-14: AI cost tracking per request (tokens + $) surfaced in metrics.
- US-15: Health check endpoint (`/health`) covering DB, Redis, queue, LLM reachability.

**Could Have (if time permits)**
- US-16: Summary customization (length / focus / tone via query params).
- US-17: Comparative analysis between two documents.
- ~~US-18~~ **Promoted to Should Have (D10):** Minimal frontend (single-page chat + upload UI) — "even basic is impressive."
- US-19: Document deletion & re-processing endpoints.

**Won't Have (out of scope — documented as assumptions)**
- Full auth/multi-tenant user management (we use a simple API-key / header-based user identity).
- Document versioning, OCR for scanned images, real-time WebSocket collaboration.

### 2.4 Confirmed Decisions
| # | Decision | Confirmed choice |
|---|---|---|
| D1 | LLM provider | **OpenAI API** (key in `.env` as `OPENAI_API_KEY`). Official `openai` Python SDK. Default chat/analysis model `gpt-4o-mini` (configurable via env; bump to `gpt-4o` for quality). JSON-schema structured outputs for document insights; SDK streaming for chat. Provider kept behind an abstraction (`LLMProvider`) so it stays swappable. *(Rev 2: switched from Claude — OpenAI provides both LLM and embeddings under one key.)* |
| D2 | Embeddings | **OpenAI `text-embedding-3-small`** (1536-dim) — same key as the LLM, cheap ($0.02/1M tokens), batched calls from the Celery worker. pgvector column: `vector(1536)`. |
| D3 | Vector store | **pgvector** inside PostgreSQL — one less moving part than Pinecone/Weaviate, earns the "vector DB integration" bonus, scales fine for 1000s of docs. |
| D4 | Task queue | **Celery + Redis**; Redis doubles as cache. Docker Desktop confirmed available → everything runs via Docker Compose. |
| D5 | File types | PDF, DOCX, TXT, MD (v1). Max upload size 25 MB. |
| D6 | Storage | **AWS S3** via `boto3` behind an abstracted `StorageBackend`. Local dev uses **MinIO** (S3-compatible) in Docker Compose; point env vars at a real S3 bucket for AWS. Pre-signed URLs available for downloads. |
| D7 | Auth | Single-tenant demo: `X-API-Key` header; `user_id` column present in schema for future multi-tenancy. |
| D8 | RAG framework | **LangChain confirmed** — used for document loaders and text splitters; RAG orchestration (retrieve → guard → prompt → cite) is hand-rolled for control and transparency. |
| D9 | Runtime | Docker Compose: `api`, `worker`, `postgres(pgvector)`, `redis`, `minio`. One-command setup. **Deployment target: single host — local Windows (Docker Desktop) now, an EC2 instance later — running the identical compose stack.** Managed AWS services (RDS/ElastiCache/ECS) are a documented future path only — see §12. |
| D10 | Frontend | **Minimal chat UI in scope** — single-page app (upload + document list + chat with citations) served as static files by FastAPI. No build toolchain. |
| D11 | Timeline | ASAP — execute Phases 0–4 sequentially without pause, then time-boxed Phase 5. |
| D12 | Demo format | Deferred — to be decided later. |

---

## 3. System Architecture

```
                         ┌──────────────────────────────────────────────┐
                         │                 FastAPI (api)                │
  Client ───────────────▶│  /documents  /chat  /metrics  /health        │
         ◀─ SSE stream ──│  Pydantic schemas · error handling · auth    │
                         └───────┬───────────────────────────┬──────────┘
                                 │ enqueue                    │ RAG query path
                                 ▼                            ▼
                         ┌──────────────┐           ┌──────────────────────┐
                         │ Redis broker │           │  RAG Service          │
                         └──────┬───────┘           │  retrieve (pgvector)  │
                                │                   │  rerank → prompt →    │
                                ▼                   │  LLM → cite → stream  │
                         ┌──────────────────────┐   └──────────┬───────────┘
                         │   Celery worker      │              │
                         │ 1. extract text      │              ▼
                         │ 2. chunk             │     ┌──────────────────┐
                         │ 3. embed (batch)     │     │  OpenAI API          │
                         │ 4. AI analysis       │     │  gpt-4o-mini +       │
                         │ 5. mark ready        │     │  text-embedding-3-sm │
                         └──────────┬───────────┘     └──────────────────────┘
                                    ▼
      ┌──────────────────────────────────────────────────────┐
      │  PostgreSQL + pgvector                               │
      │  documents · chunks(embedding) · insights ·          │
      │  chat_sessions · chat_messages · processing_metrics  │
      └──────────────────────────────────────────────────────┘

      AWS S3 via boto3 (original documents; MinIO container for local dev)
      Redis (cache: query-embedding cache, repeated-answer cache)
```

### 3.1 Processing Pipeline (async, per document)
`upload → persist file + metadata (status=queued) → Celery task:`
1. **Extract** — pdfplumber (PDF, page-aware), python-docx (DOCX), plain read (TXT/MD). Reject encrypted/empty/corrupt with precise `failure_reason`.
2. **Chunk** — recursive character splitter, ~800 tokens, 15% overlap, preserve page numbers in chunk metadata (needed for citations).
3. **Embed** — batched embedding calls; store vectors in `chunks.embedding` (pgvector, cosine).
4. **Analyze** — one structured OpenAI call → summary, key points, tags, category, entities, sentiment, language (JSON-schema structured outputs).
5. **Finalize** — status=`ready`, record stage-by-stage timings + token usage in `processing_metrics`.

Each stage is checkpointed → retries resume, failures are per-stage diagnosable.

### 3.2 RAG Chat Flow (per question)
1. Load session → scope = session's document set (must be `ready`).
2. **Condense** follow-up question into a standalone query using chat history (handles "what about clause 7?").
3. **Retrieve** top-k chunks via pgvector cosine similarity (+ optional keyword boost), filtered to scoped docs.
4. **Guard** — if max similarity < threshold → honest "not found in your documents" response (no hallucination).
5. **Generate** — grounded prompt with numbered context blocks; model must cite `[1]`, `[2]` inline; low temperature.
6. **Cite** — map markers back to doc/page/snippet; return structured `citations[]`.
7. **Persist** both turns with citations, token usage, latency; optionally stream via SSE.
8. **Suggest** 2–3 follow-up questions (cheap second call or same call, JSON).

---

## 4. Data Model (PostgreSQL + pgvector)

```
documents            id PK · user_id · filename · file_type · file_size · storage_path
                     status(queued|processing|ready|failed) · failure_reason
                     page_count · chunk_count · uploaded_at · processed_at

document_insights    id PK · document_id FK(1:1) · summary · key_points jsonb
                     tags jsonb · category · entities jsonb · sentiment
                     language · token_usage jsonb · model · created_at

chunks               id PK · document_id FK · chunk_index · content · page_number
                     token_count · embedding vector(1536)
                     [ivfflat/hnsw index, cosine]

chat_sessions        id PK · user_id · title (auto-generated) · document_ids jsonb
                     created_at · updated_at

chat_messages        id PK · session_id FK · role(user|assistant) · content
                     citations jsonb · suggested_questions jsonb
                     token_usage jsonb · latency_ms · model · created_at

processing_metrics   id PK · document_id FK · stage(extract|chunk|embed|analyze)
                     status · duration_ms · tokens_used · cost_usd · error · created_at

api_usage            id PK · endpoint · operation(embedding|chat|analysis) · model
                     prompt_tokens · completion_tokens · cost_usd · created_at
```

---

## 5. API Design (v1)

All under `/api/v1`. Consistent envelope errors: `{ "error": { "code", "message", "details" } }`.

### Documents
| Method | Path | Purpose |
|---|---|---|
| POST | `/documents/upload` | Multipart upload → `202 Accepted` + `{id, status:"queued"}` |
| GET | `/documents` | List (pagination, filter by status/tag/category, search) |
| GET | `/documents/{id}` | Metadata + status + failure_reason |
| GET | `/documents/{id}/insights` | Summary, key points, tags, entities, sentiment |
| DELETE | `/documents/{id}` | Remove doc + chunks + file *(should-have)* |

### Chat
| Method | Path | Purpose |
|---|---|---|
| POST | `/chat/sessions` | Start session `{document_ids: [...] \| null=all}` |
| GET | `/chat/sessions` | List sessions |
| POST | `/chat/sessions/{id}/messages` | Ask question → answer + citations + suggestions (`?stream=true` → SSE) |
| GET | `/chat/sessions/{id}/messages` | Chat history |

### Metrics & Ops
| Method | Path | Purpose |
|---|---|---|
| GET | `/metrics/documents` | Counts by status/type, storage used, avg chunks/doc, uploads over time |
| GET | `/metrics/processing` | Avg/p95 per-stage duration, success/failure rates, queue depth |
| GET | `/metrics/ai-usage` | Tokens & cost by operation/model/day *(bonus)* |
| GET | `/health` | DB / Redis / worker / LLM reachability |

---

## 6. Prompt Engineering Plan (documented deliverable)

All prompts versioned in `app/ai/prompts.py` with rationale comments; summarized in `AI_USAGE.md`.

1. **Document analysis prompt** — single structured-output call: summary (3–5 sentences), 5 key points, 3–7 tags, category from controlled list, entities, sentiment. JSON schema enforced; truncation strategy for long docs (map-reduce over chunk summaries when > context budget).
2. **Question condenser** — rewrite follow-up + history into standalone search query. Keeps retrieval accurate across turns.
3. **Grounded answer prompt** — rules: answer ONLY from numbered context; cite `[n]` after each claim; if insufficient context, say so explicitly; match answer language to question.
4. **Follow-up suggester** — 3 natural next questions answerable from retrieved context.
5. **Session title generator** — 4–6 word title from first exchange (UX nicety).

Anti-hallucination strategy: similarity threshold guard + "refuse if not in context" instruction + citations required → verifiable answers.

---

## 7. Project Structure (proposed)

```
app/
  main.py                 # FastAPI app factory, middleware, exception handlers
  core/                   # config (pydantic-settings), logging, security, errors
  api/v1/                 # routers: documents, chat, metrics, health
  schemas/                # Pydantic request/response models
  models/                 # SQLAlchemy ORM models
  services/               # document_service, chat_service, metrics_service
  ai/                     # llm_provider (abstract), openai_provider, prompts, rag/
    rag/                  # retriever, context_builder, citation_mapper, guard
  processing/             # extractors (pdf/docx/txt), chunker, pipeline stages
  workers/                # celery_app, tasks
  db/                     # session, migrations (alembic)
tests/                    # pytest: unit (mocked LLM) + integration (API-level)
docker-compose.yml        # api, worker, postgres+pgvector, redis
Docs/                     # this plan, architecture notes, API examples
README.md · AI_USAGE.md · .env.example
```

---

## 8. Execution Phases & Task Breakdown

### Phase 0 — Foundation (scaffold, ~first session)
- [ ] Repo scaffold, FastAPI app factory, settings, structured logging, error handlers
- [ ] Docker Compose (postgres+pgvector, redis, api, worker), Alembic baseline migration
- [ ] Health endpoint, CI-friendly `pytest` setup

### Phase 1 — Document Ingestion (Must)
- [ ] Upload endpoint: validation (type, size, empty), storage abstraction, `202 + id`
- [ ] Celery pipeline: extract → chunk → embed → status transitions, per-stage metrics, retry/failure handling
- [ ] Document list/detail endpoints
- [ ] Tests: extractor edge cases (corrupt, encrypted, empty), pipeline with mocked LLM

### Phase 2 — AI Insights (Must)
- [ ] LLM provider abstraction + OpenAI implementation (`openai` SDK) + token/cost accounting
- [ ] Analysis stage (structured JSON insights) + insights endpoint
- [ ] Map-reduce summarization fallback for very large documents

### Phase 3 — RAG Chat (Must — the core demo)
- [ ] Sessions CRUD; pgvector retriever with doc scoping
- [ ] Question condenser → retrieve → guard → grounded answer → citation mapping
- [ ] Multi-turn history windowing; persist messages with citations & usage
- [ ] "Not in documents" honest-refusal path
- [ ] Tests: retrieval relevance, citation integrity, multi-turn behavior (mocked LLM)

### Phase 4 — Metrics & Hardening (Must)
- [ ] Documents / processing / AI-usage metrics endpoints
- [ ] Rate limiting (simple middleware), request logging, consistent error catalog
- [ ] README (setup, architecture, assumptions) + AI_USAGE.md + API examples (curl/Postman)

### Phase 4.5 — Minimal Chat UI (committed per D10)
- [ ] Single static page served by FastAPI: upload with progress/status polling, document list with insights, chat panel with citations + follow-up suggestions
- [ ] No build toolchain — vanilla HTML/CSS/JS (or a single CDN-free file)

### Phase 5 — Bonus (in value order, time-boxed)
- [ ] SSE streaming answers (API + UI)
- [ ] Follow-up question suggestions
- [ ] Multi-document chat (already enabled by session scoping — polish + test)
- [ ] Redis caching (query-embedding cache, repeated-answer cache)
- [ ] Summary customization params; comparative analysis endpoint

**Demo script (deliverable #9):** upload 2–3 sample docs → watch status progress → fetch insights → chat with citations → multi-turn follow-up → "not in document" question → metrics endpoints.

---

## 9. Decision Log (resolved 2026-10-13)

| Question | Resolution |
|---|---|
| LLM provider & key | **OpenAI API** (rev 2, 2026-10-13) — key in `.env` as `OPENAI_API_KEY`. Initially Claude was chosen, but Claude has no embeddings API; OpenAI covers chat (`gpt-4o-mini`) **and** embeddings (`text-embedding-3-small`) with one key. |
| Cloud services | **AWS** — S3 for document storage now (boto3 + MinIO locally); full AWS deployment mapping in §12 for the rest of the stack. |
| Docker | **Docker Desktop available** → full Docker Compose stack (api, worker, postgres+pgvector, redis, minio). |
| LangChain depth | **Confirmed** — LangChain for loaders/splitters; hand-rolled RAG orchestration. |
| Frontend | **Minimal chat UI in scope** — single static page served by FastAPI (upload + chat with citations). |
| Timeline | **ASAP** — execute phases back-to-back; Phase 5 strictly time-boxed. |
| Demo format | **Deferred** — decide after core build. |

---

## 10. Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| LLM API cost/latency during dev | Slow iteration, spend | `gpt-4o-mini` default (cheap), mocked LLM/embeddings in tests, Redis response cache, batched embedding calls |
| Celery on Windows | Dev friction | Run everything in Docker Compose (decision D4/D9) |
| Hallucinated answers in demo | Credibility | Similarity guard + mandatory citations + refusal path (US-09) |
| Large PDFs blow context limits | Analysis failures | Chunk-level map-reduce summarization, token budgeting |
| Scope creep into bonus items | Unfinished core | Phases 1–4 locked first; Phase 5 strictly time-boxed |
| Corrupt/exotic files | Pipeline crashes | Per-stage try/except, `failed` status with precise reason, retry endpoint |

---

## 11. Evaluation-Criteria Mapping (how this plan scores)

| Criterion | Where we earn it |
|---|---|
| AI-First (33%) | Hand-tuned prompt suite (§6), structured-output insights, condensed-question RAG, anti-hallucination guard, follow-up suggestions, AI_USAGE.md documenting Claude Code usage |
| Product Thinking (33%) | MoSCoW prioritization, 202-async UX, honest-refusal answers, precise failure reasons, intuitive REST design, assumptions documented |
| Technical (33%) | Clean layered architecture, Celery pipeline with per-stage checkpoints, pgvector, Alembic migrations, pytest with mocked LLM, Docker one-command setup, OpenAPI docs |

---

## 12. Deployment: Local → EC2 (now), Managed AWS (later)

**Committed deployment model (rev 3):** the app runs as a self-contained Docker Compose stack on a **single host** — the local Windows machine during development, then an **EC2 instance** for hosting. Deploying to EC2 is: install Docker + Compose, clone the repo, copy `.env`, `docker compose up -d`. Nothing in the stack assumes a managed AWS service.

EC2 specifics we will bake in:
- All services bind behind the compose network; only the API port (and optionally MinIO console) is exposed — front with a security group (and nginx/Caddy for TLS if made public).
- Data durability via named Docker volumes (`pgdata`, `minio-data`) so restarts/redeploys keep documents and vectors.
- `restart: unless-stopped` on all services; `.env`-driven config so the same image works on both hosts.
- Instance sizing note: `t3.large` (2 vCPU / 8 GB) is comfortable for the full stack + embedding batches.

**Future scaling path** (documented for the "Think About Scale" criterion — configuration changes only, not build work):

| Compose service (local & EC2) | Managed AWS equivalent | Notes |
|---|---|---|
| MinIO | **S3** | Same `boto3` code path — only endpoint/credentials env vars change. `StorageBackend` is S3-native from day one, so switching to real S3 is an `.env` edit. |
| postgres + pgvector container | **RDS for PostgreSQL** (pgvector supported) | Alembic migrations run unchanged. |
| redis container | **ElastiCache (Redis)** | Broker + cache. (Alternative: SQS as Celery broker if ElastiCache is overkill.) |
| api container | **ECS Fargate** (or App Runner) behind an ALB | Same image. |
| worker container | **ECS Fargate** service | Scale workers independently of the API. |
| `.env` secrets | **Secrets Manager / SSM Parameter Store** | `OPENAI_API_KEY`, DB URL, S3 bucket. |
| Logs | **CloudWatch Logs** | Structured JSON logs ship as-is. |

**Committed for this build:** the single-host compose stack (local + EC2) with MinIO as the S3-compatible store. The managed-services table is a documented scaling path, not build work — it goes in the README's architecture section.

