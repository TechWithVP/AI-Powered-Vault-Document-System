"""API-key authentication."""

from __future__ import annotations

import pytest

from app.core.config import AppSettings, Settings
from app.core.exceptions import AuthenticationError
from app.core.security import ApiKeyAuthenticator, Principal


class _StubSettings(Settings):
    """Settings double exposing only the app section the authenticator reads."""

    def __init__(self, api_keys: str) -> None:  # noqa: D107 - test double
        self.app = AppSettings(api_keys=api_keys)


@pytest.fixture
def authenticator() -> ApiKeyAuthenticator:
    return ApiKeyAuthenticator(_StubSettings("valid-key,second-key"))


class TestApiKeyAuthenticator:
    def test_known_key_resolves_to_a_principal(
        self, authenticator: ApiKeyAuthenticator
    ) -> None:
        assert authenticator.authenticate("valid-key") == Principal(
            user_id="valid-key"
        )

    def test_any_configured_key_is_accepted(
        self, authenticator: ApiKeyAuthenticator
    ) -> None:
        assert authenticator.authenticate("second-key").user_id == "second-key"

    def test_missing_key_is_rejected(
        self, authenticator: ApiKeyAuthenticator
    ) -> None:
        with pytest.raises(AuthenticationError, match="Missing"):
            authenticator.authenticate(None)

    def test_empty_key_is_rejected(
        self, authenticator: ApiKeyAuthenticator
    ) -> None:
        with pytest.raises(AuthenticationError):
            authenticator.authenticate("")

    def test_unknown_key_is_rejected(
        self, authenticator: ApiKeyAuthenticator
    ) -> None:
        with pytest.raises(AuthenticationError, match="not valid"):
            authenticator.authenticate("attacker-key")

    def test_key_prefix_is_not_accepted(
        self, authenticator: ApiKeyAuthenticator
    ) -> None:
        with pytest.raises(AuthenticationError):
            authenticator.authenticate("valid")
