"""Configuration section behaviour."""

from __future__ import annotations

import pytest

from app.core.config import (
    AppSettings,
    PostgresSettings,
    RedisSettings,
    S3Settings,
    StorageBackendType,
    get_settings,
)


class TestAppSettings:
    def test_api_keys_are_split_and_trimmed(self) -> None:
        settings = AppSettings(api_keys=" key-one , key-two ,, ")
        assert settings.allowed_api_keys == frozenset({"key-one", "key-two"})

    def test_production_flag_follows_environment(self) -> None:
        assert AppSettings(env="production").is_production is True
        assert AppSettings(env="local").is_production is False


class TestPostgresSettings:
    def test_dsn_includes_credentials_and_database(self) -> None:
        dsn = PostgresSettings(
            host="db", port=6543, db="vault", user="u", password="p"
        ).async_dsn
        assert dsn == "postgresql+psycopg://u:p@db:6543/vault"


class TestRedisSettings:
    def test_broker_and_cache_use_distinct_databases(self) -> None:
        settings = RedisSettings(host="r", port=6379, broker_db=0, cache_db=1)
        assert settings.broker_url.endswith("/0")
        assert settings.cache_url.endswith("/1")

    def test_password_is_embedded_when_present(self) -> None:
        assert RedisSettings(password="secret").broker_url.startswith(
            "redis://:secret@"
        )


class TestS3Settings:
    @pytest.mark.parametrize("blank", ["", "   "])
    def test_blank_endpoint_selects_real_aws(self, blank: str) -> None:
        settings = S3Settings(endpoint_url=blank)
        assert settings.endpoint_url is None
        assert settings.is_custom_endpoint is False

    def test_explicit_endpoint_is_treated_as_custom(self) -> None:
        assert S3Settings(endpoint_url="http://localstack:4566").is_custom_endpoint


class TestSettingsComposition:
    def test_settings_is_a_cached_singleton(self) -> None:
        assert get_settings() is get_settings()

    def test_every_section_is_present(self) -> None:
        settings = get_settings()
        for section in (
            "app",
            "postgres",
            "redis",
            "storage",
            "s3",
            "openai",
            "ingestion",
            "rag",
        ):
            assert hasattr(settings, section), section

    def test_storage_backend_defaults_to_local(self) -> None:
        assert get_settings().storage.backend in set(StorageBackendType)
