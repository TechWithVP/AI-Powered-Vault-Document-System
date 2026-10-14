"""Document API contract tests.

The service layer is replaced with a stub so these exercise routing, auth,
serialisation and error mapping without needing Postgres or Celery.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from http import HTTPStatus
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.deps import get_document_service
from app.core.constants import DocumentStatus, FileType
from app.core.exceptions import DocumentNotReadyError, NotFoundError
from app.models.document import Document

DOCUMENT_ID = uuid.uuid4()


def make_document(status: DocumentStatus = DocumentStatus.READY) -> Document:
    return Document(
        id=DOCUMENT_ID,
        user_id="test-api-key",
        filename="policy.pdf",
        file_type=FileType.PDF,
        file_size_bytes=2048,
        storage_key="documents/abc/policy.pdf",
        status=status,
        page_count=3,
        chunk_count=7,
        character_count=900,
        created_at=datetime.now(tz=UTC),
        updated_at=datetime.now(tz=UTC),
    )


@pytest.fixture
def service() -> MagicMock:
    stub = MagicMock()
    stub.upload = AsyncMock(return_value=make_document(DocumentStatus.QUEUED))
    stub.get = AsyncMock(return_value=make_document())
    stub.list = AsyncMock(return_value=([make_document()], 1))
    stub.delete = AsyncMock(return_value=None)
    return stub


@pytest.fixture
def api(app: FastAPI, service: MagicMock) -> TestClient:
    app.dependency_overrides[get_document_service] = lambda: service
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


class TestAuthentication:
    def test_requests_without_a_key_are_rejected(self, api: TestClient) -> None:
        response = api.get("/api/v1/documents")
        assert response.status_code == HTTPStatus.UNAUTHORIZED
        assert response.json()["error"]["code"] == "authentication_error"

    def test_requests_with_an_unknown_key_are_rejected(
        self, api: TestClient
    ) -> None:
        response = api.get(
            "/api/v1/documents", headers={"X-API-Key": "not-a-real-key"}
        )
        assert response.status_code == HTTPStatus.UNAUTHORIZED


class TestUpload:
    def test_accepted_upload_returns_202_and_queues_processing(
        self, api: TestClient, auth_headers: dict[str, str], service: MagicMock
    ) -> None:
        with patch("app.api.v1.documents.process_document") as task:
            response = api.post(
                "/api/v1/documents/upload",
                headers=auth_headers,
                files={"file": ("policy.pdf", b"%PDF-1.4", "application/pdf")},
            )

        assert response.status_code == HTTPStatus.ACCEPTED
        body = response.json()
        assert body["status"] == DocumentStatus.QUEUED
        assert body["id"] == str(DOCUMENT_ID)
        task.apply_async.assert_called_once()
        service.upload.assert_awaited_once()

    def test_an_already_processed_duplicate_is_not_requeued(
        self, api: TestClient, auth_headers: dict[str, str], service: MagicMock
    ) -> None:
        service.upload = AsyncMock(return_value=make_document(DocumentStatus.READY))

        with patch("app.api.v1.documents.process_document") as task:
            response = api.post(
                "/api/v1/documents/upload",
                headers=auth_headers,
                files={"file": ("policy.pdf", b"content", "application/pdf")},
            )

        assert response.status_code == HTTPStatus.ACCEPTED
        task.apply_async.assert_not_called()

    def test_a_missing_file_field_is_a_validation_error(
        self, api: TestClient, auth_headers: dict[str, str]
    ) -> None:
        response = api.post("/api/v1/documents/upload", headers=auth_headers)
        assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
        assert response.json()["error"]["code"] == "validation_error"


class TestRetrieval:
    def test_detail_returns_document_metadata(
        self, api: TestClient, auth_headers: dict[str, str]
    ) -> None:
        response = api.get(f"/api/v1/documents/{DOCUMENT_ID}", headers=auth_headers)

        assert response.status_code == HTTPStatus.OK
        body = response.json()
        assert body["filename"] == "policy.pdf"
        assert body["page_count"] == 3
        assert body["chunk_count"] == 7
        assert body["character_count"] == 900

    def test_unknown_document_returns_the_error_envelope(
        self, api: TestClient, auth_headers: dict[str, str], service: MagicMock
    ) -> None:
        service.get = AsyncMock(
            side_effect=NotFoundError.for_resource("Document", DOCUMENT_ID)
        )
        response = api.get(f"/api/v1/documents/{DOCUMENT_ID}", headers=auth_headers)

        assert response.status_code == HTTPStatus.NOT_FOUND
        assert response.json()["error"]["code"] == "not_found"

    def test_a_malformed_id_is_rejected(
        self, api: TestClient, auth_headers: dict[str, str]
    ) -> None:
        response = api.get("/api/v1/documents/not-a-uuid", headers=auth_headers)
        assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY

    def test_list_returns_a_page_envelope(
        self, api: TestClient, auth_headers: dict[str, str]
    ) -> None:
        response = api.get("/api/v1/documents", headers=auth_headers)

        assert response.status_code == HTTPStatus.OK
        body = response.json()
        assert body["total"] == 1
        assert len(body["items"]) == 1
        assert {"items", "total", "limit", "offset"} <= body.keys()

    def test_list_filters_are_passed_to_the_service(
        self, api: TestClient, auth_headers: dict[str, str], service: MagicMock
    ) -> None:
        api.get(
            "/api/v1/documents?status=ready&tag=policy&search=remote&limit=5",
            headers=auth_headers,
        )

        criteria, params = service.list.await_args.args
        assert criteria.status is DocumentStatus.READY
        assert criteria.tag == "policy"
        assert criteria.search == "remote"
        assert params.limit == 5

    def test_an_invalid_status_filter_is_rejected(
        self, api: TestClient, auth_headers: dict[str, str]
    ) -> None:
        response = api.get("/api/v1/documents?status=nonsense", headers=auth_headers)
        assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY


class TestInsights:
    def test_missing_insights_return_404_with_the_status(
        self, api: TestClient, auth_headers: dict[str, str], service: MagicMock
    ) -> None:
        service.get = AsyncMock(return_value=make_document(DocumentStatus.PROCESSING))

        response = api.get(
            f"/api/v1/documents/{DOCUMENT_ID}/insights", headers=auth_headers
        )

        assert response.status_code == HTTPStatus.NOT_FOUND
        assert response.json()["error"]["details"]["status"] == "processing"


class TestDeletion:
    def test_delete_returns_no_content(
        self, api: TestClient, auth_headers: dict[str, str], service: MagicMock
    ) -> None:
        response = api.delete(f"/api/v1/documents/{DOCUMENT_ID}", headers=auth_headers)

        assert response.status_code == HTTPStatus.NO_CONTENT
        service.delete.assert_awaited_once()


class TestNotReadyDocuments:
    def test_conflict_is_mapped_to_409(
        self, api: TestClient, auth_headers: dict[str, str], service: MagicMock
    ) -> None:
        service.get = AsyncMock(
            side_effect=DocumentNotReadyError.for_documents([str(DOCUMENT_ID)])
        )
        response = api.get(f"/api/v1/documents/{DOCUMENT_ID}", headers=auth_headers)

        assert response.status_code == HTTPStatus.CONFLICT
        assert response.json()["error"]["code"] == "document_not_ready"
