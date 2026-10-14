"""Document persistence."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy import Select, func, or_, select
from sqlalchemy.dialects.postgresql import INTERVAL
from sqlalchemy.orm import selectinload

from app.core.constants import DocumentCategory, DocumentStatus
from app.models.document import Document, DocumentInsight
from app.repositories.base import AsyncRepository, SyncRepository


@dataclass(frozen=True, slots=True)
class DocumentFilter:
    """Query criteria for listing documents."""

    user_id: str
    status: DocumentStatus | None = None
    category: DocumentCategory | None = None
    tag: str | None = None
    search: str | None = None


class DocumentQueryMixin:
    """Builds the filtered document query shared by both repositories."""

    @staticmethod
    def build_query(criteria: DocumentFilter) -> Select[tuple[Document]]:
        statement = select(Document).where(Document.user_id == criteria.user_id)

        if criteria.status is not None:
            statement = statement.where(Document.status == criteria.status)

        if criteria.category is not None or criteria.tag is not None:
            statement = statement.join(Document.insight)
            if criteria.category is not None:
                statement = statement.where(
                    DocumentInsight.category == criteria.category
                )
            if criteria.tag is not None:
                # Containment against the JSONB tag array uses the GIN-friendly
                # @> operator rather than unnesting every row.
                statement = statement.where(
                    DocumentInsight.tags.contains([criteria.tag])
                )

        if criteria.search:
            pattern = f"%{criteria.search.strip()}%"
            statement = statement.outerjoin(Document.insight).where(
                or_(
                    Document.filename.ilike(pattern),
                    DocumentInsight.summary.ilike(pattern),
                )
            )

        return statement.order_by(Document.created_at.desc())


class DocumentRepository(AsyncRepository[Document], DocumentQueryMixin):
    """Async document queries used by the API."""

    model = Document

    async def get_for_user(
        self, document_id: uuid.UUID, user_id: str
    ) -> Document | None:
        """Fetch a document scoped to its owner, eagerly loading insights."""
        statement = (
            select(Document)
            .where(Document.id == document_id, Document.user_id == user_id)
            .options(selectinload(Document.insight))
        )
        return await self.session.scalar(statement)

    async def list_page(
        self, criteria: DocumentFilter, *, limit: int, offset: int
    ) -> tuple[list[Document], int]:
        """Return one page of documents plus the unpaginated total."""
        statement = self.build_query(criteria).options(
            selectinload(Document.insight)
        )
        total = await self.count(statement)
        page = await self.list(statement.limit(limit).offset(offset))
        return page, total

    async def get_many_for_user(
        self, document_ids: list[uuid.UUID], user_id: str
    ) -> list[Document]:
        """Fetch several documents at once, preserving owner scoping."""
        if not document_ids:
            return []
        statement = select(Document).where(
            Document.id.in_(document_ids), Document.user_id == user_id
        )
        return await self.list(statement)

    async def find_by_content_hash(
        self, content_hash: str, user_id: str
    ) -> Document | None:
        """Detect a re-upload of identical bytes by the same user."""
        statement = select(Document).where(
            Document.content_hash == content_hash,
            Document.user_id == user_id,
            Document.status != DocumentStatus.FAILED,
        )
        return await self.session.scalar(statement.limit(1))

    async def count_by_status(self, user_id: str | None = None) -> dict[str, int]:
        """Aggregate document counts per status for the metrics endpoint."""
        statement = select(Document.status, func.count()).group_by(Document.status)
        if user_id is not None:
            statement = statement.where(Document.user_id == user_id)
        rows = await self.session.execute(statement)
        return {str(status): count for status, count in rows.all()}


class SyncDocumentRepository(SyncRepository[Document]):
    """Sync document access used by the ingestion worker."""

    model = Document

    def get_or_fail(self, document_id: uuid.UUID) -> Document:
        from app.core.exceptions import NotFoundError

        document = self.get(document_id)
        if document is None:
            raise NotFoundError.for_resource("Document", document_id)
        return document

    def upsert_insight(self, insight: DocumentInsight) -> DocumentInsight:
        """Replace any existing analysis so re-processing stays idempotent."""
        existing = self.session.scalar(
            select(DocumentInsight).where(
                DocumentInsight.document_id == insight.document_id
            )
        )
        if existing is not None:
            self.session.delete(existing)
            self.session.flush()
        self.session.add(insight)
        return insight


class DocumentStatsRepository(AsyncRepository[Document]):
    """Aggregate document queries backing the statistics endpoint."""

    model = Document

    async def counts_by_status(self, user_id: str) -> dict[str, int]:
        statement = (
            select(Document.status, func.count())
            .where(Document.user_id == user_id)
            .group_by(Document.status)
        )
        rows = await self.session.execute(statement)
        return {str(status): int(count) for status, count in rows.all()}

    async def counts_by_file_type(self, user_id: str) -> dict[str, int]:
        statement = (
            select(Document.file_type, func.count())
            .where(Document.user_id == user_id)
            .group_by(Document.file_type)
        )
        rows = await self.session.execute(statement)
        return {str(file_type): int(count) for file_type, count in rows.all()}

    async def counts_by_category(self, user_id: str) -> dict[str, int]:
        statement = (
            select(DocumentInsight.category, func.count())
            .join(Document, Document.id == DocumentInsight.document_id)
            .where(Document.user_id == user_id)
            .group_by(DocumentInsight.category)
        )
        rows = await self.session.execute(statement)
        return {str(category): int(count) for category, count in rows.all()}

    async def totals(self, user_id: str) -> dict[str, float]:
        """Corpus-wide totals and averages across the user's documents."""
        statement = select(
            func.count().label("documents"),
            func.coalesce(func.sum(Document.file_size_bytes), 0).label("bytes"),
            func.coalesce(func.sum(Document.chunk_count), 0).label("chunks"),
            func.coalesce(func.sum(Document.page_count), 0).label("pages"),
            func.coalesce(func.avg(Document.chunk_count), 0).label("avg_chunks"),
        ).where(Document.user_id == user_id)

        row = (await self.session.execute(statement)).one()
        return {
            "documents": int(row.documents or 0),
            "storage_bytes": int(row.bytes or 0),
            "chunks": int(row.chunks or 0),
            "pages": int(row.pages or 0),
            "avg_chunks_per_document": round(float(row.avg_chunks or 0), 2),
        }

    async def top_tags(self, user_id: str, *, limit: int = 10) -> list[dict[str, int]]:
        """Most frequent AI-assigned tags, for a sense of corpus composition."""
        tag = func.jsonb_array_elements_text(DocumentInsight.tags).column_valued("tag")
        statement = (
            select(tag, func.count().label("count"))
            .select_from(DocumentInsight)
            .join(Document, Document.id == DocumentInsight.document_id)
            .where(Document.user_id == user_id)
            .group_by(tag)
            .order_by(func.count().desc())
            .limit(limit)
        )
        rows = await self.session.execute(statement)
        return [{"tag": name, "count": int(count)} for name, count in rows.all()]

    async def uploads_by_day(self, user_id: str, *, days: int = 14) -> list[dict]:
        """Upload volume per day, oldest first."""
        day = func.date_trunc("day", Document.created_at).label("day")
        cutoff = func.now() - func.cast(f"{days} days", INTERVAL)
        statement = (
            select(day, func.count().label("count"))
            .where(Document.user_id == user_id, Document.created_at >= cutoff)
            .group_by(day)
            .order_by(day)
        )
        rows = await self.session.execute(statement)
        return [
            {"date": value.date().isoformat(), "count": int(count)}
            for value, count in rows.all()
        ]
