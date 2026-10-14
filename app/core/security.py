"""API-key authentication.

The case study is single-tenant, so identity is a shared API key sent on the
X-API-Key header. Resolving it to a Principal here keeps every downstream layer
user-scoped, so swapping in real multi-tenant auth later touches only this
module.
"""

from __future__ import annotations

import hmac
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Security
from fastapi.security import APIKeyHeader

from app.core.config import Settings, get_settings
from app.core.exceptions import AuthenticationError

API_KEY_HEADER_NAME = "X-API-Key"

_api_key_header = APIKeyHeader(
    name=API_KEY_HEADER_NAME,
    auto_error=False,
    description="Shared API key issued to the client.",
)


@dataclass(frozen=True, slots=True)
class Principal:
    """The authenticated caller. Every stored row is scoped to its user_id."""

    user_id: str

    @classmethod
    def from_api_key(cls, api_key: str) -> "Principal":
        # The key doubles as the tenant identifier in this single-tenant build.
        return cls(user_id=api_key)


class ApiKeyAuthenticator:
    """Validates API keys in constant time against the configured allow-list."""

    def __init__(self, settings: Settings) -> None:
        self._allowed = settings.app.allowed_api_keys

    def authenticate(self, api_key: str | None) -> Principal:
        if not api_key:
            raise AuthenticationError(
                f"Missing {API_KEY_HEADER_NAME} header.",
                details={"header": API_KEY_HEADER_NAME},
            )
        if not self._is_known(api_key):
            raise AuthenticationError("The provided API key is not valid.")
        return Principal.from_api_key(api_key)

    def _is_known(self, candidate: str) -> bool:
        # compare_digest on every candidate avoids leaking key length via timing.
        return any(hmac.compare_digest(candidate, known) for known in self._allowed)


def get_authenticator(
    settings: Annotated[Settings, Depends(get_settings)],
) -> ApiKeyAuthenticator:
    return ApiKeyAuthenticator(settings)


def get_principal(
    api_key: Annotated[str | None, Security(_api_key_header)],
    authenticator: Annotated[ApiKeyAuthenticator, Depends(get_authenticator)],
) -> Principal:
    """FastAPI dependency resolving the caller, or raising AuthenticationError."""
    return authenticator.authenticate(api_key)


CurrentPrincipal = Annotated[Principal, Depends(get_principal)]
