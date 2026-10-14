"""Document chat endpoints."""

from __future__ import annotations

import json
import uuid
from http import HTTPStatus

from collections.abc import AsyncIterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.api.deps import (
    AppSettingsDep,
    CacheDep,
    ChatServiceDep,
    CurrentPrincipal,
    EmbedderDep,
    LLMDep,
    Pagination,
    StorageDep,
    build_chat_service,
)
from app.db.session import db_manager
from app.core.exceptions import AppError
from app.core.logging import get_logger
from app.schemas.chat import (
    AnswerResponse,
    AskRequest,
    CreateSessionRequest,
    MessageResponse,
    SessionResponse,
)
from app.schemas.common import Page

logger = get_logger(__name__)

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post(
    "/sessions",
    status_code=HTTPStatus.CREATED,
    response_model=SessionResponse,
    summary="Start a chat session",
    description=(
        "Opens a conversation scoped to one or more processed documents. Omit "
        "document_ids to chat across every document you own. Documents that "
        "are still processing are rejected with 409."
    ),
)
async def create_session(
    payload: CreateSessionRequest,
    principal: CurrentPrincipal,
    service: ChatServiceDep,
) -> SessionResponse:
    session = await service.create_session(
        principal, document_ids=payload.document_ids, title=payload.title
    )
    return SessionResponse.from_model(session)


@router.get(
    "/sessions",
    response_model=Page[SessionResponse],
    summary="List chat sessions",
    description="Returns your chat sessions, most recently active first.",
)
async def list_sessions(
    principal: CurrentPrincipal,
    service: ChatServiceDep,
    params: Pagination,
) -> Page[SessionResponse]:
    sessions, total = await service.list_sessions(
        principal, limit=params.limit, offset=params.offset
    )
    return Page.build(
        [SessionResponse.from_model(session) for session in sessions],
        total=total,
        params=params,
    )


@router.get(
    "/sessions/{session_id}",
    response_model=SessionResponse,
    summary="Get a chat session",
)
async def get_session(
    session_id: uuid.UUID,
    principal: CurrentPrincipal,
    service: ChatServiceDep,
) -> SessionResponse:
    session = await service.get_session(session_id, principal)
    return SessionResponse.from_model(session)


@router.post(
    "/sessions/{session_id}/messages",
    response_model=AnswerResponse,
    summary="Ask a question",
    description=(
        "Answers a question using only the session's documents. The answer "
        "cites the passages it relies on; when the documents do not cover the "
        "question the response says so and grounded is false."
    ),
)
async def ask_question(
    session_id: uuid.UUID,
    payload: AskRequest,
    principal: CurrentPrincipal,
    service: ChatServiceDep,
) -> AnswerResponse:
    result = await service.ask(
        session_id,
        principal,
        payload.question,
        suggest_follow_ups=payload.suggest_follow_ups,
    )
    return AnswerResponse.from_result(result)


@router.post(
    "/sessions/{session_id}/messages/stream",
    summary="Ask a question (streaming)",
    description=(
        "Same as asking a question, but the answer is streamed as "
        "Server-Sent Events. Emits `token` events as text is generated, then "
        "a `citations` event once sources are resolved, then `done`. "
        "Citations arrive at the end because they can only be determined after "
        "the model has finished citing."
    ),
    response_class=StreamingResponse,
    responses={
        200: {
            "content": {"text/event-stream": {}},
            "description": "Server-Sent Event stream",
        }
    },
)
async def ask_question_streaming(
    session_id: uuid.UUID,
    payload: AskRequest,
    principal: CurrentPrincipal,
    settings: AppSettingsDep,
    storage: StorageDep,
    llm: LLMDep,
    embedder: EmbedderDep,
    cache: CacheDep,
) -> StreamingResponse:
    async def event_source() -> AsyncIterator[str]:
        # The generator runs after the endpoint returns, so it opens its own
        # unit of work; a request-scoped session would already be closed and
        # the conversation would never be persisted.
        try:
            async with db_manager.session() as session:
                service = build_chat_service(
                    session,
                    settings=settings,
                    storage=storage,
                    llm=llm,
                    embedder=embedder,
                    cache=cache,
                )
                async for event in service.ask_stream(
                    session_id, principal, payload.question
                ):
                    yield _format_sse(event.type.value, event.data)
        except AppError as exc:
            # The response has already begun, so the error is delivered as a
            # terminal event rather than an HTTP status.
            yield _format_sse("error", exc.to_payload()["error"])
        except Exception:
            logger.exception("Streaming answer failed")
            yield _format_sse(
                "error",
                {"code": "internal_error", "message": "The answer stream failed."},
            )

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            # Prevents buffering by reverse proxies, which would defeat streaming.
            "X-Accel-Buffering": "no",
        },
    )


def _format_sse(event: str, data: dict) -> str:
    """Render one Server-Sent Event frame."""
    return (
        f"event: {event}\n"
        f"data: {json.dumps(data)}\n\n"
    )


@router.get(
    "/sessions/{session_id}/messages",
    response_model=Page[MessageResponse],
    summary="Get chat history",
    description="Returns the conversation in chronological order.",
)
async def list_messages(
    session_id: uuid.UUID,
    principal: CurrentPrincipal,
    service: ChatServiceDep,
    params: Pagination,
) -> Page[MessageResponse]:
    messages, total = await service.list_messages(
        session_id, principal, limit=params.limit, offset=params.offset
    )
    return Page.build(
        [MessageResponse.from_model(message) for message in messages],
        total=total,
        params=params,
    )


@router.delete(
    "/sessions/{session_id}",
    status_code=HTTPStatus.NO_CONTENT,
    response_model=None,
    summary="Delete a chat session",
)
async def delete_session(
    session_id: uuid.UUID,
    principal: CurrentPrincipal,
    service: ChatServiceDep,
) -> None:
    await service.delete_session(session_id, principal)
