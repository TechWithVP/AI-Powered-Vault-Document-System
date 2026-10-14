# AI Usage

How AI was used to build this system, and how AI is used inside it.

---

## Part 1: Building the system

The project was built with **Claude Code** (Anthropic's CLI) driving an
agentic workflow: analysing requirements, writing code, running it against the
live Docker stack, and fixing what broke.

### The workflow that worked

**Plan before code.** The requirements PDF was turned into a written project
plan (`Docs/ProjectPlan.md`) with user stories under MoSCoW prioritisation, a
data model, an API surface, and six execution phases. Three decisions changed
during review - the provider moved from Claude to OpenAI once it became clear
Claude has no embeddings API, storage moved to S3-compatible, and the chat UI
was promoted from optional to in-scope. Those were cheap changes on paper and
would have been expensive mid-build.

**Phase by phase, each verified before moving on.** Foundation, ingestion,
insights, chat, metrics, UI. Each phase ended with the stack running and its
behaviour checked against the real API, not just against tests.

**Verify against the running system, not the model's confidence.** Every claim
in this repository was checked by executing it: documents uploaded through the
live API, chunks inspected in Postgres, embeddings confirmed as 1536-dimension
vectors, costs read back from the metrics endpoint. Several defects were only
visible this way.

### Defects found by running the code

These are a fair sample of why execution matters more than review:

| Defect | Symptom | Fix |
|---|---|---|
| `extra={"filename": ...}` collided with a reserved `LogRecord` attribute | Every upload returned a 500 | A `ContextLogger` that nests caller fields under one key, so the whole class of collision cannot recur |
| Enum columns loaded as plain strings | `'str' object has no attribute 'value'` in the worker | A `StrEnumType` decorator that round-trips enums |
| The chunk-size filter discarded short documents entirely | A one-line document failed ingestion with "no usable text chunks" | Filter fragments only when substantive chunks survive; otherwise keep what there is |
| A named Docker volume mounted root-owned | Storage health check failed as the non-root user | Create the directory in the image so the volume inherits ownership |
| `os.environ.setdefault` in tests | Compose exports `.env` into the container, so tests used the developer's real API key | Force test values instead of defaulting |
| `python-docx` raises `BadZipFile`, not `PackageNotFoundError`, for non-zip input | Users got "File is not a zip file" instead of guidance | Catch both and explain that `.doc` must be saved as `.docx` |

### Where AI needed correction

- **Default-shaped code.** First drafts reached for plausible-looking APIs that
  do not exist (`func.cast(..., func.INTEGER())`) or patterns that are merely
  conventional. Compiling and running caught these; reading did not.
- **Tests written to pass.** When a test failed, the tempting fix was to relax
  the assertion. In the two cases where that came up - short-document chunking
  and the refusal path - the test was pointing at a real design question, and
  the right fix was in the application code. One test was rewritten to assert
  the actual invariant (the model is never asked to answer from weak context)
  rather than a proxy for it (call count).
- **Unprompted scope.** Generated code tended toward extra abstraction layers.
  The rule applied throughout was that an interface earns its place only if
  there is a second implementation or a test double behind it.

---

## Measuring retrieval instead of trusting it

Every retrieval constant here - chunk size, overlap, the similarity and lexical
thresholds, fusion weights - started as a number that looked reasonable while
hand-checking a few questions. That is not tuning, and it hid three real
defects that only appeared once a labelled set existed:

1. **Fusion could not promote an exact-term hit.** With both arms truncated to
   `top_k` and the lexical arm weighted at 0.8, a chunk only full-text search
   found scored below the weakest semantic hit and never reached the model. The
   two candidate lists were entirely disjoint at that depth, so reciprocal rank
   fusion had nothing to fuse and degenerated into concatenation. Fixed by
   fetching a deeper candidate pool and weighting the arms equally; hybrid
   recall@6 went from 82 percent - identical to semantic alone - to 91.
2. **Concurrent queries on one session.** The two searches ran under
   `asyncio.gather` on the request's shared `AsyncSession`, which SQLAlchemy
   does not permit. It never surfaced in production only because earlier
   queries in the request had already provisioned the connection. They now run
   sequentially; two index lookups are noise against an LLM call.
3. **Lexical-only citations reported `similarity: 0.0`**, advertising a correct
   citation as irrelevant. Citations now carry both scores.

The harness in `eval/` reports each arm separately rather than one blended
number, because the first defect was invisible in the aggregate: hybrid and
semantic scored identically while the lexical arm contributed nothing at all.

## Part 2: AI inside the system

Four prompts do the work, all in `app/ai/prompts.py` with the reasoning for
each recorded next to it.

### Principles applied to every prompt

1. **Describe success, not failure.** Models follow a description of the
   desired output more reliably than a list of prohibitions.
2. **Make refusal a first-class outcome.** The answerer is told that "the
   documents do not cover this" is a correct and useful answer. Without that,
   a model under pressure to be helpful will fill the gap.
3. **Let the schema carry the format.** Structured outputs enforce shape, so
   prompts spend their words on judgement instead of JSON instructions.
4. **Stable content first.** The invariant preamble precedes the variable
   content so repeated calls share a cacheable prefix.

### 1. Document analysis

Runs once per document and feeds search, filtering and the document list.

The core instruction is to stay strictly inside the text: *"If the document
does not establish something, omit it rather than inferring it - a plausible
guess is worse than a shorter answer, because downstream search treats your
output as fact."* An invented tag silently corrupts retrieval for every later
query, so the cost of a hallucination here is persistent rather than one-off.

Summaries are framed for *a colleague deciding whether this document is
relevant to them*, which produces substance rather than a restatement of the
title. Key points are specified as the statements someone would need to act on,
which keeps them concrete rather than structural ("the document has five
sections").

Output is constrained by a strict JSON schema, so there is no parse-and-repair
path.

### 2. Question condensation

Follow-ups like "And what about contractors?" embed poorly on their own - the
subject lives in the previous turn. Before retrieval, the question is rewritten
into a standalone form using the conversation.

The prompt tells the model to keep the user's own terminology, because that
usually matches the document's wording, which is what the index contains.
Paraphrasing into synonyms measurably hurts retrieval. The call is skipped on
the first turn, and a failure falls back to the raw question rather than failing
the request.

### 3. Grounded answering

The trust boundary of the system. Three things are mandatory:

- **Citations after every claim**, so the answer is verifiable rather than
  merely plausible.
- **Only the supplied excerpts** as source material.
- **An explicit escape hatch**: when the excerpts do not answer the question,
  say so and describe what they do cover.

This is backed by code rather than trusted to the prompt. Retrieval that clears
neither the semantic nor the lexical threshold never reaches the model at all -
the system answers honestly without spending a token. After generation, the
cited markers are mapped back to real excerpts and renumbered; an answer citing
nothing valid is flagged `grounded: false` in the response.

A caveat is added for counting and listing questions. Retrieval returns a sample
of the document, so a total computed from it cannot be trusted; the prompt
requires the model to say that its figure covers only the excerpts shown. This
was added after a real session answered "there are 41 exclusions" from three
fragments of a 76-chunk policy, with citations that made the number look
verified.

### 4. Conversation questions

Questions about the exchange itself - "what did I ask?", "where did that excerpt
come from?" - are answerable from the transcript, which is already in memory.
Routing them through document retrieval produced a refusal every time, which
reads as broken rather than careful. They are now detected before retrieval and
answered from the transcript, with `source: conversation` on the response so a
client can distinguish them from document-grounded answers.

### 5. Follow-up suggestions

Suggestions are rendered as clickable prompts, so they must be answerable from
the same excerpts. Suggesting a question the vault cannot answer trains users to
distrust the feature, so the prompt constrains suggestions to the retrieved
material.

---

## Cost

Measured on the running system with `gpt-4o-mini` and
`text-embedding-3-small`:

| Operation | Typical cost |
|---|---|
| Ingesting a 3-page document (embed + analyse) | about $0.00025 |
| One chat question (query embedding + answer) | about $0.00008 |

Every provider call is recorded in `api_usage` as it happens, with token counts
and USD cost, and is queryable at `/api/v1/metrics/ai-usage`. Costs are
calculated from published rates in `app/ai/pricing.py`; an unknown model prices
at zero rather than failing an ingestion run.

Cost control measures built in:

- `gpt-4o-mini` by default, with the model configurable per environment.
- Embeddings batched 64 chunks per request.
- Duplicate uploads short-circuit on content hash.
- The condensing call is skipped on the first turn of a conversation.
- Below-threshold retrieval skips answer generation entirely.
