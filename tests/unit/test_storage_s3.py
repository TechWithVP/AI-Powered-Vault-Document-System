"""S3 storage backend: key handling and error translation.

boto3 is replaced with a stub, so these verify the behaviour the application
depends on - which AWS errors mean "missing" versus "broken" - without needing
AWS credentials or a LocalStack container.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from botocore.exceptions import ClientError, EndpointConnectionError

from app.core.config import S3Settings
from app.core.exceptions import NotFoundError, StorageError
from app.storage.s3 import S3StorageBackend

BUCKET = "vault-documents"


def client_error(code: str) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": code}}, "GetObject")


@pytest.fixture
def settings() -> S3Settings:
    return S3Settings(
        bucket=BUCKET,
        region="us-east-1",
        endpoint_url="http://localstack:4566",
        access_key_id="test",
        secret_access_key="test",
        use_ssl=False,
    )


@pytest.fixture
def backend(settings: S3Settings) -> S3StorageBackend:
    instance = S3StorageBackend(settings)
    instance._client = MagicMock()
    return instance


class TestSave:
    def test_the_object_is_put_with_its_content_type(
        self, backend: S3StorageBackend
    ) -> None:
        stored = backend.save("docs/a.pdf", b"bytes", content_type="application/pdf")

        kwargs = backend.client.put_object.call_args.kwargs
        assert kwargs["Bucket"] == BUCKET
        assert kwargs["Key"] == "docs/a.pdf"
        assert kwargs["ContentType"] == "application/pdf"
        assert stored.size_bytes == 5

    def test_a_put_failure_becomes_a_storage_error(
        self, backend: S3StorageBackend
    ) -> None:
        backend.client.put_object.side_effect = client_error("AccessDenied")

        with pytest.raises(StorageError, match="Failed to store"):
            backend.save("docs/a.pdf", b"bytes", content_type="application/pdf")

    def test_a_network_failure_becomes_a_storage_error(
        self, backend: S3StorageBackend
    ) -> None:
        backend.client.put_object.side_effect = EndpointConnectionError(
            endpoint_url="http://localstack:4566"
        )

        with pytest.raises(StorageError):
            backend.save("docs/a.pdf", b"bytes", content_type="application/pdf")


class TestLoad:
    def test_object_bytes_are_returned(self, backend: S3StorageBackend) -> None:
        body = MagicMock()
        body.read.return_value = b"file contents"
        backend.client.get_object.return_value = {"Body": body}

        assert backend.load("docs/a.pdf") == b"file contents"

    @pytest.mark.parametrize("code", ["NoSuchKey", "404", "NotFound"])
    def test_a_missing_object_is_not_found_not_an_outage(
        self, backend: S3StorageBackend, code: str
    ) -> None:
        """Distinguishing missing from broken is what lets the API return 404."""
        backend.client.get_object.side_effect = client_error(code)

        with pytest.raises(NotFoundError):
            backend.load("docs/missing.pdf")

    def test_other_aws_errors_remain_storage_errors(
        self, backend: S3StorageBackend
    ) -> None:
        backend.client.get_object.side_effect = client_error("InternalError")

        with pytest.raises(StorageError):
            backend.load("docs/a.pdf")


class TestExists:
    def test_a_present_object_is_reported(self, backend: S3StorageBackend) -> None:
        assert backend.exists("docs/a.pdf") is True

    def test_a_missing_object_is_reported_without_raising(
        self, backend: S3StorageBackend
    ) -> None:
        backend.client.head_object.side_effect = client_error("404")
        assert backend.exists("docs/a.pdf") is False

    def test_an_unexpected_error_still_raises(
        self, backend: S3StorageBackend
    ) -> None:
        backend.client.head_object.side_effect = client_error("AccessDenied")

        with pytest.raises(StorageError):
            backend.exists("docs/a.pdf")


class TestDelete:
    def test_the_object_is_deleted(self, backend: S3StorageBackend) -> None:
        backend.delete("docs/a.pdf")

        kwargs = backend.client.delete_object.call_args.kwargs
        assert kwargs == {"Bucket": BUCKET, "Key": "docs/a.pdf"}

    def test_a_delete_failure_is_reported(self, backend: S3StorageBackend) -> None:
        backend.client.delete_object.side_effect = client_error("AccessDenied")

        with pytest.raises(StorageError):
            backend.delete("docs/a.pdf")


class TestPresignedUrls:
    def test_a_url_is_generated_with_the_default_ttl(
        self, backend: S3StorageBackend, settings: S3Settings
    ) -> None:
        backend.client.generate_presigned_url.return_value = "https://signed"

        assert backend.presigned_url("docs/a.pdf") == "https://signed"
        kwargs = backend.client.generate_presigned_url.call_args.kwargs
        assert kwargs["ExpiresIn"] == settings.presigned_url_ttl_seconds

    def test_the_ttl_can_be_overridden(self, backend: S3StorageBackend) -> None:
        backend.presigned_url("docs/a.pdf", expires_in=60)
        assert backend.client.generate_presigned_url.call_args.kwargs["ExpiresIn"] == 60


class TestBucketProvisioning:
    def test_an_existing_bucket_is_left_alone(
        self, backend: S3StorageBackend
    ) -> None:
        backend.ensure_bucket()
        backend.client.create_bucket.assert_not_called()

    def test_a_missing_bucket_is_created(self, backend: S3StorageBackend) -> None:
        backend.client.head_bucket.side_effect = client_error("404")

        backend.ensure_bucket()

        backend.client.create_bucket.assert_called_once_with(Bucket=BUCKET)

    def test_losing_a_create_race_is_not_an_error(
        self, backend: S3StorageBackend
    ) -> None:
        """Another worker may have created it first; that is benign."""
        backend.client.head_bucket.side_effect = client_error("404")
        backend.client.create_bucket.side_effect = client_error(
            "BucketAlreadyOwnedByYou"
        )

        backend.ensure_bucket()

    def test_a_real_create_failure_is_raised(
        self, backend: S3StorageBackend
    ) -> None:
        backend.client.head_bucket.side_effect = client_error("404")
        backend.client.create_bucket.side_effect = client_error("AccessDenied")

        with pytest.raises(StorageError, match="Cannot create bucket"):
            backend.ensure_bucket()


class TestHealthCheck:
    def test_a_reachable_bucket_passes(self, backend: S3StorageBackend) -> None:
        backend.health_check()

    def test_an_unreachable_bucket_fails(self, backend: S3StorageBackend) -> None:
        backend.client.head_bucket.side_effect = EndpointConnectionError(
            endpoint_url="http://localstack:4566"
        )

        with pytest.raises(StorageError, match="unhealthy"):
            backend.health_check()


class TestClientConstruction:
    def test_path_addressing_is_used_for_compatibility(
        self, settings: S3Settings
    ) -> None:
        """Virtual-host addressing does not work against MinIO or LocalStack."""
        client = S3StorageBackend(settings)._build_client()
        assert client.meta.config.s3["addressing_style"] == "path"

    def test_the_custom_endpoint_is_honoured(self, settings: S3Settings) -> None:
        client = S3StorageBackend(settings)._build_client()
        assert "localstack" in client.meta.endpoint_url
