"""ORM models.

Importing this package registers every model on ``Base.metadata``, which is what
Alembic autogeneration and ``create_all`` rely on.
"""

from app.models.chat import ChatMessage, ChatSession, chat_session_documents
from app.models.chunk import EMBEDDING_DIMENSIONS, DocumentChunk
from app.models.document import Document, DocumentInsight
from app.models.metrics import ApiUsage, ProcessingMetric

__all__ = [
    "EMBEDDING_DIMENSIONS",
    "ApiUsage",
    "ChatMessage",
    "ChatSession",
    "Document",
    "DocumentChunk",
    "DocumentInsight",
    "ProcessingMetric",
    "chat_session_documents",
]
