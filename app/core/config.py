"""Application configuration.

Settings are grouped into cohesive, independently testable sections, each
backed by its own environment prefix. ``Settings`` composes the sections and is
exposed through the cached :func:`get_settings` accessor so the whole process
shares a single immutable configuration object.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import Field, computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


def _section_config(env_prefix: str) -> SettingsConfigDict:
    """Build the settings config for a section with its own env prefix.

    Every section reads the shared ``.env`` file and ignores variables owned by
    other sections, so unrelated keys never raise.
    """
    return SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
        env_prefix=env_prefix,
    )


class _Section(BaseSettings):
    """Base class for configuration sections."""

    model_config = _section_config("")


class Environment(StrEnum):
    LOCAL = "local"
    STAGING = "staging"
    PRODUCTION = "production"


class AppSettings(_Section):
    """Process-level settings: identity, logging and authentication."""

    model_config = _section_config("APP_")

    name: str = "AI-Powered Vault Document System"
    version: str = "1.0.0"
    env: Environment = Environment.LOCAL
    debug: bool = False
    log_level: str = "INFO"
    log_json: bool = True
    api_keys: str = "dev-key-change-me"

    @property
    def is_production(self) -> bool:
        return self.env is Environment.PRODUCTION

    @computed_field  # type: ignore[prop-decorator]
    @property
    def allowed_api_keys(self) -> frozenset[str]:
        """API keys accepted on the ``X-API-Key`` header."""
        return frozenset(k.strip() for k in self.api_keys.split(",") if k.strip())


class PostgresSettings(_Section):
    """PostgreSQL connection settings (pgvector-enabled instance)."""

    model_config = _section_config("POSTGRES_")

    host: str = "postgres"
    port: int = 5432
    db: str = "vault"
    user: str = "vault"
    password: str = "vault"
    pool_size: int = 10
    max_overflow: int = 20
    echo: bool = False

    def _dsn(self, driver: str) -> str:
        return (
            f"postgresql+{driver}://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.db}"
        )

    @property
    def async_dsn(self) -> str:
        """DSN for the FastAPI request path (async engine)."""
        return self._dsn("psycopg")

    @property
    def sync_dsn(self) -> str:
        """DSN for Celery workers and Alembic (sync engine)."""
        return self._dsn("psycopg")


class RedisSettings(_Section):
    """Redis settings: Celery broker/result backend and the response cache."""

    model_config = _section_config("REDIS_")

    host: str = "redis"
    port: int = 6379
    broker_db: int = 0
    cache_db: int = 1
    password: str | None = None

    def _url(self, db: int) -> str:
        auth = f":{self.password}@" if self.password else ""
        return f"redis://{auth}{self.host}:{self.port}/{db}"

    @property
    def broker_url(self) -> str:
        return self._url(self.broker_db)

    @property
    def cache_url(self) -> str:
        return self._url(self.cache_db)


class StorageBackendType(StrEnum):
    """Selects which StorageBackend implementation is used."""

    LOCAL = "local"
    S3 = "s3"


class StorageSettings(_Section):
    """Chooses and configures the document storage backend."""

    model_config = _section_config("STORAGE_")

    backend: StorageBackendType = StorageBackendType.LOCAL
    local_path: Path = Path("/srv/app/data/documents")

    @property
    def uses_s3(self) -> bool:
        return self.backend is StorageBackendType.S3


class S3Settings(_Section):
    """S3-compatible object storage (AWS S3, or LocalStack for development)."""

    model_config = _section_config("S3_")

    bucket: str = "vault-documents"
    region: str = "us-east-1"
    endpoint_url: str | None = None
    access_key_id: str | None = None
    secret_access_key: str | None = None
    use_ssl: bool = True
    presigned_url_ttl_seconds: int = 900

    @field_validator("endpoint_url", mode="before")
    @classmethod
    def _blank_endpoint_means_aws(cls, value: object) -> object:
        """An empty endpoint selects real AWS S3 rather than a custom host."""
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @property
    def is_custom_endpoint(self) -> bool:
        return self.endpoint_url is not None


class OpenAISettings(_Section):
    """OpenAI client settings for chat completions and embeddings."""

    model_config = _section_config("OPENAI_")

    api_key: str = ""
    chat_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    timeout_seconds: float = 60.0
    max_retries: int = 3
    temperature: float = 0.1

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)


class IngestionSettings(_Section):
    """Upload limits and the extract/chunk/embed pipeline tuning knobs."""

    model_config = _section_config("INGEST_")

    max_upload_mb: int = 25
    chunk_tokens: int = 350
    chunk_overlap_ratio: float = Field(default=0.18, ge=0.0, lt=0.5)
    embed_batch_size: int = 64
    max_analysis_chars: int = 120_000

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def chunk_overlap_tokens(self) -> int:
        return int(self.chunk_tokens * self.chunk_overlap_ratio)


class RateLimitSettings(_Section):
    """Per-client request limiting."""

    model_config = _section_config("RATE_LIMIT_")

    enabled: bool = True
    requests: int = 120
    window_seconds: int = 60


class RagSettings(_Section):
    """Retrieval-augmented generation tuning."""

    model_config = _section_config("RAG_")

    top_k: int = 6
    min_similarity: float = Field(default=0.25, ge=0.0, le=1.0)
    #: ts_rank floor for accepting purely lexical evidence. plainto_tsquery
    #: already requires every query term to be present, so this only screens
    #: out incidental matches.
    min_lexical_rank: float = Field(default=0.02, ge=0.0)
    #: Counting and listing questions need a wider view of the document.
    aggregate_top_k: int = 12
    #: Candidates fetched from each retrieval arm before fusion. Fusion rewards
    #: chunks both rankers agree on, so the lists must be deep enough to
    #: overlap; truncating each to top_k leaves nothing to fuse and silently
    #: degrades the hybrid into semantic search with a lexical tail.
    candidate_pool: int = Field(default=40, ge=1)
    history_turns: int = 6
    max_context_chars: int = 24_000
    suggestion_count: int = 3


class Settings:
    """Root configuration object composing every settings section."""

    def __init__(self) -> None:
        self.app = AppSettings()
        self.postgres = PostgresSettings()
        self.redis = RedisSettings()
        self.storage = StorageSettings()
        self.s3 = S3Settings()
        self.openai = OpenAISettings()
        self.ingestion = IngestionSettings()
        self.rag = RagSettings()
        self.rate_limit = RateLimitSettings()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()
