"""Tests for abtem.dataerai._client: SDK upload / relationship layer.

The real ``dataerai`` SDK talks to a local daemon; these tests substitute a
fake ``dataerai`` module via ``sys.modules`` so every degradation path is
exercised without network, daemon, or the SDK installed.
"""

import sys
import types
from dataclasses import dataclass
from typing import Optional

import pytest

from abtem.dataerai._client import LinkOutcome, PreservationClient, UploadOutcome
from abtem.dataerai._config import DataeraiConfig


@dataclass
class FakeUploadResult:
    asset_id: str = "aaaaaaaa-bbbb-cccc-dddd-eeeeffff0001"
    content_id: str = "aaaaaaaa-bbbb-cccc-dddd-eeeeffff0002"
    transfer_id: str = "aaaaaaaa-bbbb-cccc-dddd-eeeeffff0003"
    total_bytes: int = 3
    chunk_count: int = 1


@dataclass
class FakeAuthStatus:
    user_email: str = "user@example.org"
    expires_at: str = "2100-01-01T00:00:00Z"
    user_id: str = "99999999-8888-7777-6666-555544443333"


class FakeDaemonError(Exception):
    def __init__(self, code, message=""):
        super().__init__(message or code)
        self.code = code


class FakeClient:
    """Stands in for dataerai.DataeraiClient; records calls.

    Mirrors the published SDK contract: requests fail until ``connect()``
    is called.
    """

    upload_error: Optional[Exception] = None
    relationship_error: Optional[Exception] = None
    auth_error: Optional[Exception] = None
    connect_error: Optional[Exception] = None
    instances: list = []

    def __init__(self, *args, **kwargs):
        self.uploads = []
        self.relationships = []
        self.connected = False
        self.closed = False
        FakeClient.instances.append(self)

    def connect(self):
        if FakeClient.connect_error is not None:
            raise FakeClient.connect_error
        self.connected = True

    def close(self):
        self.connected = False
        self.closed = True

    def _require_connected(self):
        if not self.connected:
            raise RuntimeError("Not connected — call connect() first")

    def auth_status(self):
        self._require_connected()
        if FakeClient.auth_error is not None:
            raise FakeClient.auth_error
        return FakeAuthStatus()

    def upload(self, local_path, **kwargs):
        self._require_connected()
        if FakeClient.upload_error is not None:
            raise FakeClient.upload_error
        self.uploads.append((local_path, kwargs))
        return FakeUploadResult()

    def create_relationship(self, from_asset_id, to_asset_id, rel_type, **kwargs):
        self._require_connected()
        if FakeClient.relationship_error is not None:
            raise FakeClient.relationship_error
        self.relationships.append((from_asset_id, to_asset_id, rel_type, kwargs))
        return types.SimpleNamespace(id="rel-1", type=rel_type)


@pytest.fixture
def fake_sdk(monkeypatch):
    module = types.ModuleType("dataerai")
    module.DataeraiClient = FakeClient
    module.DaemonError = FakeDaemonError
    monkeypatch.setitem(sys.modules, "dataerai", module)
    FakeClient.instances = []
    FakeClient.upload_error = None
    FakeClient.relationship_error = None
    FakeClient.auth_error = None
    FakeClient.connect_error = None
    yield module


@pytest.fixture
def no_sdk(monkeypatch):
    monkeypatch.setitem(sys.modules, "dataerai", None)  # forces ImportError


@pytest.fixture
def payload(tmp_path):
    path = tmp_path / "artifact.json"
    path.write_text("{}")
    return path


def live_config(**kwargs):
    kwargs.setdefault("dry_run", False)
    return DataeraiConfig(**kwargs)


class TestUpload:
    def test_dry_run_skips_without_touching_sdk(self, no_sdk, payload):
        client = PreservationClient(DataeraiConfig(dry_run=True))

        outcome = client.upload(payload, title="t")

        assert outcome == UploadOutcome(status="skipped", detail="dry-run")

    def test_missing_sdk_skips(self, no_sdk, payload):
        client = PreservationClient(live_config())

        outcome = client.upload(payload, title="t")

        assert outcome.status == "skipped"
        assert "SDK" in outcome.detail

    def test_project_owner_upload(self, fake_sdk, payload):
        project_id = "12121212-3434-5656-7878-909090909090"
        client = PreservationClient(live_config(project_id=project_id))

        outcome = client.upload(
            payload,
            title="structure",
            record_type="sample_specimen",
            tags=["abtem"],
            metadata={"run_id": "r1"},
            description="d",
        )

        assert outcome.status == "uploaded"
        assert outcome.asset_id == FakeUploadResult.asset_id
        (local_path, kwargs) = FakeClient.instances[0].uploads[0]
        assert local_path == str(payload)
        assert kwargs["title"] == "structure"
        assert kwargs["owner_type"] == "project"
        assert kwargs["owner_id"] == project_id
        assert kwargs["record_type"] == "sample_specimen"
        assert kwargs["tags"] == ["abtem"]
        assert kwargs["metadata"] == {"run_id": "r1"}
        assert kwargs["description"] == "d"

    def test_user_owner_fallback(self, fake_sdk, payload):
        client = PreservationClient(live_config())

        outcome = client.upload(payload, title="t")

        assert outcome.status == "uploaded"
        (_, kwargs) = FakeClient.instances[0].uploads[0]
        assert kwargs["owner_type"] == "user"
        assert kwargs["owner_id"] == FakeAuthStatus.user_id

    def test_auth_failure_fails(self, fake_sdk, payload):
        FakeClient.auth_error = RuntimeError("daemon not running")
        client = PreservationClient(live_config())

        outcome = client.upload(payload, title="t")

        assert outcome.status == "failed"
        assert "daemon not running" in outcome.detail

    def test_upload_error_fails(self, fake_sdk, payload):
        FakeClient.upload_error = RuntimeError("quota exceeded")
        client = PreservationClient(live_config(project_id="p" * 8))

        outcome = client.upload(payload, title="t")

        assert outcome.status == "failed"
        assert "quota exceeded" in outcome.detail

    def test_owner_resolved_once(self, fake_sdk, payload):
        client = PreservationClient(live_config())

        client.upload(payload, title="a")
        client.upload(payload, title="b")

        # one client instance, one auth_status round-trip, two uploads
        assert len(FakeClient.instances) == 1
        assert len(FakeClient.instances[0].uploads) == 2

    def test_connects_published_sdk_before_use(self, fake_sdk, payload):
        client = PreservationClient(live_config())

        outcome = client.upload(payload, title="t")

        assert outcome.status == "uploaded"
        assert FakeClient.instances[0].connected is True

    def test_connect_failure_degrades(self, fake_sdk, payload):
        FakeClient.connect_error = RuntimeError("daemon unreachable")
        client = PreservationClient(live_config())

        outcome = client.upload(payload, title="t")

        assert outcome.status == "failed"
        assert "daemon unreachable" in outcome.detail
        # and stays failed (no crash) on subsequent calls
        assert client.upload(payload, title="t2").status == "failed"

    def test_close_closes_sdk_client(self, fake_sdk, payload):
        client = PreservationClient(live_config())
        client.upload(payload, title="t")

        client.close()

        assert FakeClient.instances[0].closed is True

    def test_close_without_sdk_is_safe(self, no_sdk):
        client = PreservationClient(live_config())

        client.close()  # must not raise

    def test_sdk_without_connect_still_works(self, monkeypatch, payload):
        class EagerClient:  # SDK generation that connects implicitly, no close()
            def __init__(self, *args, **kwargs):
                self.uploads = []

            def auth_status(self):
                return FakeAuthStatus()

            def upload(self, local_path, **kwargs):
                self.uploads.append((local_path, kwargs))
                return FakeUploadResult()

        module = types.ModuleType("dataerai")
        module.DataeraiClient = EagerClient
        monkeypatch.setitem(sys.modules, "dataerai", module)
        client = PreservationClient(live_config())

        outcome = client.upload(payload, title="t")

        assert outcome.status == "uploaded"
        client.close()  # no close() on the SDK client: still safe


class TestLink:
    FROM = "aaaaaaaa-0000-0000-0000-000000000001"
    TO = "aaaaaaaa-0000-0000-0000-000000000002"

    def test_dry_run_skips(self, no_sdk):
        client = PreservationClient(DataeraiConfig(dry_run=True))

        outcome = client.link(self.FROM, self.TO, "derived_from")

        assert outcome == LinkOutcome(status="skipped", detail="dry-run")

    def test_sdk_link_created(self, fake_sdk):
        client = PreservationClient(live_config())

        outcome = client.link(
            self.FROM,
            self.TO,
            "acquired_with",
            qualifiers={"role": "scan"},
            analysis_mode="non_destructive",
            qualifier_note="note",
        )

        assert outcome.status == "created"
        (from_id, to_id, rel_type, kwargs) = FakeClient.instances[0].relationships[0]
        assert (from_id, to_id, rel_type) == (self.FROM, self.TO, "acquired_with")
        assert kwargs["qualifiers"] == {"role": "scan"}
        assert kwargs["analysis_mode"] == "non_destructive"
        assert kwargs["qualifier_note"] == "note"

    def test_existing_relationship_reported(self, fake_sdk):
        FakeClient.relationship_error = FakeDaemonError("ERR_RELATIONSHIP_EXISTS")
        client = PreservationClient(live_config())

        outcome = client.link(self.FROM, self.TO, "derived_from")

        assert outcome.status == "exists"

    def test_sdk_error_fails(self, fake_sdk):
        FakeClient.relationship_error = FakeDaemonError(
            "ERR_PERMISSION_DENIED", "no write scope"
        )
        client = PreservationClient(live_config())

        outcome = client.link(self.FROM, self.TO, "derived_from")

        assert outcome.status == "failed"
        assert "no write scope" in outcome.detail

    def test_old_sdk_falls_back_to_rest(self, monkeypatch):
        class OldClient:  # an SDK predating create_relationship
            def __init__(self, *args, **kwargs):
                pass

            def auth_status(self):
                return FakeAuthStatus()

        module = types.ModuleType("dataerai")
        module.DataeraiClient = OldClient
        monkeypatch.setitem(sys.modules, "dataerai", module)

        calls = []

        def fake_post(url, payload, token, timeout=30.0):
            calls.append((url, payload, token))
            return 201, {"id": "rel-9"}

        monkeypatch.setattr("abtem.dataerai._client._post_json", fake_post)
        client = PreservationClient(
            live_config(server="https://beta.example.org", token="tok-1")
        )

        outcome = client.link(
            self.FROM, self.TO, "derived_from", qualifiers={"step": "1"}
        )

        assert outcome.status == "created"
        url, body, token = calls[0]
        assert url == (
            f"https://beta.example.org/api/assets/{self.FROM}/relationships/"
        )
        assert body["to_asset_id"] == self.TO
        assert body["type"] == "derived_from"
        assert body["qualifiers"] == {"step": "1"}
        assert token == "tok-1"

    def test_rest_used_when_no_sdk(self, no_sdk, monkeypatch):
        calls = []

        def fake_post(url, payload, token, timeout=30.0):
            calls.append(url)
            return 201, {"id": "rel-10"}

        monkeypatch.setattr("abtem.dataerai._client._post_json", fake_post)
        client = PreservationClient(live_config(token="tok-2"))

        outcome = client.link(self.FROM, self.TO, "analysis_of")

        assert outcome.status == "created"
        assert len(calls) == 1

    def test_rest_409_means_exists(self, no_sdk, monkeypatch):
        monkeypatch.setattr(
            "abtem.dataerai._client._post_json",
            lambda url, payload, token, timeout=30.0: (409, {}),
        )
        client = PreservationClient(live_config(token="tok"))

        outcome = client.link(self.FROM, self.TO, "derived_from")

        assert outcome.status == "exists"

    def test_rest_error_fails(self, no_sdk, monkeypatch):
        monkeypatch.setattr(
            "abtem.dataerai._client._post_json",
            lambda url, payload, token, timeout=30.0: (403, {"detail": "forbidden"}),
        )
        client = PreservationClient(live_config(token="tok"))

        outcome = client.link(self.FROM, self.TO, "derived_from")

        assert outcome.status == "failed"

    def test_no_sdk_no_token_skips(self, no_sdk):
        client = PreservationClient(live_config())

        outcome = client.link(self.FROM, self.TO, "derived_from")

        assert outcome.status == "skipped"


class TestPostJson:
    def test_posts_bearer_json(self, monkeypatch):
        from abtem.dataerai import _client

        captured = {}

        class FakeResponse:
            status = 201

            def read(self):
                return b'{"id": "rel-1"}'

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        def fake_urlopen(request, timeout=None):
            captured["url"] = request.full_url
            captured["auth"] = request.get_header("Authorization")
            captured["content_type"] = request.get_header("Content-type")
            captured["body"] = request.data
            return FakeResponse()

        monkeypatch.setattr(_client, "urlopen", fake_urlopen)

        status, body = _client._post_json(
            "https://s.example/api/x/", {"a": 1}, "tok", timeout=5.0
        )

        assert status == 201
        assert body == {"id": "rel-1"}
        assert captured["auth"] == "Bearer tok"
        assert captured["content_type"] == "application/json"
        assert b'"a": 1' in captured["body"]

    def test_http_error_returns_status(self, monkeypatch):
        import urllib.error

        from abtem.dataerai import _client

        def fake_urlopen(request, timeout=None):
            raise urllib.error.HTTPError(
                request.full_url, 409, "conflict", None, None
            )

        monkeypatch.setattr(_client, "urlopen", fake_urlopen)

        status, body = _client._post_json("https://s.example/api/x/", {}, "tok")

        assert status == 409
