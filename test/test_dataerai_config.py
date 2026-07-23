"""Tests for abtem.dataerai configuration and environment snapshotting."""

import json
import subprocess
import sys

import pytest

from abtem.dataerai._config import (
    DEFAULT_SERVER,
    DataeraiConfig,
    _credentials_file_token,
    _keychain_token,
    discover_token,
)
from abtem.dataerai._environment import environment_snapshot


@pytest.fixture
def no_ambient_credentials(monkeypatch):
    """Isolate tests from real tokens/SDK installed on the developer machine."""
    monkeypatch.setattr("abtem.dataerai._config._keychain_token", lambda: None)
    monkeypatch.setattr(
        "abtem.dataerai._config._credentials_file_token", lambda path=None: None
    )
    monkeypatch.setattr("abtem.dataerai._config._sdk_available", lambda: False)


class TestFromEnv:
    def test_defaults_offline(self, no_ambient_credentials):
        config = DataeraiConfig.from_env(environ={})

        assert config.server == DEFAULT_SERVER
        assert config.project_id is None
        assert config.owner_type == "project"
        assert config.token is None
        assert config.dry_run is True

    def test_env_overrides(self, no_ambient_credentials):
        config = DataeraiConfig.from_env(
            environ={
                "DATAERAI_SERVER": "https://example.org/",
                "DATAERAI_PROJECT_ID": "6f2a2c1e-aaaa-bbbb-cccc-000000000001",
                "DATAERAI_OWNER_TYPE": "user",
            }
        )

        assert config.server == "https://example.org"  # trailing slash stripped
        assert config.project_id == "6f2a2c1e-aaaa-bbbb-cccc-000000000001"
        assert config.owner_type == "user"

    def test_token_from_env_enables_live_mode(self, no_ambient_credentials):
        config = DataeraiConfig.from_env(environ={"DATAERAI_TOKEN": "tok-123"})

        assert config.token == "tok-123"
        assert config.dry_run is False

    def test_sdk_available_enables_live_mode(self, no_ambient_credentials, monkeypatch):
        monkeypatch.setattr("abtem.dataerai._config._sdk_available", lambda: True)

        config = DataeraiConfig.from_env(environ={})

        assert config.dry_run is False

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes"])
    def test_dry_run_forced_on_even_with_token(
        self, no_ambient_credentials, value
    ):
        config = DataeraiConfig.from_env(
            environ={"DATAERAI_TOKEN": "tok", "DATAERAI_DRY_RUN": value}
        )

        assert config.dry_run is True

    @pytest.mark.parametrize("value", ["0", "false", "no"])
    def test_dry_run_forced_off_without_token(self, no_ambient_credentials, value):
        config = DataeraiConfig.from_env(environ={"DATAERAI_DRY_RUN": value})

        assert config.dry_run is False

    def test_invalid_owner_type_raises(self, no_ambient_credentials):
        with pytest.raises(ValueError, match="owner_type"):
            DataeraiConfig.from_env(environ={"DATAERAI_OWNER_TYPE": "collection"})

    def test_token_never_in_repr(self):
        config = DataeraiConfig(token="secret-token-value", dry_run=False)

        assert "secret-token-value" not in repr(config)


class TestTokenDiscovery:
    def test_env_token_wins_over_other_sources(self, monkeypatch):
        monkeypatch.setattr(
            "abtem.dataerai._config._keychain_token", lambda: "from-keychain"
        )
        monkeypatch.setattr(
            "abtem.dataerai._config._credentials_file_token",
            lambda path=None: "from-file",
        )

        assert discover_token({"DATAERAI_TOKEN": "from-env"}) == "from-env"

    def test_keychain_wins_over_file(self, monkeypatch):
        monkeypatch.setattr(
            "abtem.dataerai._config._keychain_token", lambda: "from-keychain"
        )
        monkeypatch.setattr(
            "abtem.dataerai._config._credentials_file_token",
            lambda path=None: "from-file",
        )

        assert discover_token({}) == "from-keychain"

    def test_file_used_last(self, monkeypatch):
        monkeypatch.setattr("abtem.dataerai._config._keychain_token", lambda: None)
        monkeypatch.setattr(
            "abtem.dataerai._config._credentials_file_token",
            lambda path=None: "from-file",
        )

        assert discover_token({}) == "from-file"

    def test_no_sources_returns_none(self, monkeypatch):
        monkeypatch.setattr("abtem.dataerai._config._keychain_token", lambda: None)
        monkeypatch.setattr(
            "abtem.dataerai._config._credentials_file_token", lambda path=None: None
        )

        assert discover_token({}) is None

    def test_credentials_file_env_override(self, monkeypatch, tmp_path):
        credentials = tmp_path / "creds.json"
        credentials.write_text(json.dumps({"access_token": "override-tok"}))
        monkeypatch.setattr("abtem.dataerai._config._keychain_token", lambda: None)

        token = discover_token({"DATAERAI_CREDENTIALS_FILE": str(credentials)})

        assert token == "override-tok"


class TestKeychainToken:
    def test_reads_token_on_darwin(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "darwin")

        def fake_run(cmd, **kwargs):
            assert "security" in cmd[0]
            return subprocess.CompletedProcess(cmd, 0, stdout="tok-from-keychain\n")

        monkeypatch.setattr(subprocess, "run", fake_run)

        assert _keychain_token() == "tok-from-keychain"

    def test_json_blob_in_keychain(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "darwin")
        blob = json.dumps({"access_token": "tok-json", "refresh_token": "r"})

        def fake_run(cmd, **kwargs):
            return subprocess.CompletedProcess(cmd, 0, stdout=blob + "\n")

        monkeypatch.setattr(subprocess, "run", fake_run)

        assert _keychain_token() == "tok-json"

    def test_hex_encoded_json_blob_in_keychain(self, monkeypatch):
        # macOS `security -w` prints hex when the stored value is not
        # plain ASCII; the daemon stores a credentials JSON blob.
        monkeypatch.setattr(sys, "platform", "darwin")
        blob = json.dumps({"access_token": "tok-hex", "user_email": "u@x.é"})
        hex_blob = blob.encode("utf-8").hex()

        def fake_run(cmd, **kwargs):
            return subprocess.CompletedProcess(cmd, 0, stdout=hex_blob + "\n")

        monkeypatch.setattr(subprocess, "run", fake_run)

        assert _keychain_token() == "tok-hex"

    def test_json_blob_without_token_yields_none(self, monkeypatch):
        # a structured blob with no usable token must not be sent as a
        # bearer header
        monkeypatch.setattr(sys, "platform", "darwin")
        blob = json.dumps({"refresh_token": "r", "user_email": "u@x"})

        def fake_run(cmd, **kwargs):
            return subprocess.CompletedProcess(cmd, 0, stdout=blob + "\n")

        monkeypatch.setattr(subprocess, "run", fake_run)

        assert _keychain_token() is None

    def test_short_hexlike_token_kept_verbatim(self, monkeypatch):
        # a real token that merely looks hex-ish must not be mangled
        monkeypatch.setattr(sys, "platform", "darwin")

        def fake_run(cmd, **kwargs):
            return subprocess.CompletedProcess(cmd, 0, stdout="deadbeef\n")

        monkeypatch.setattr(subprocess, "run", fake_run)

        assert _keychain_token() == "deadbeef"

    def test_returns_none_off_darwin(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")

        assert _keychain_token() is None

    def test_returns_none_when_missing(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "darwin")

        def fake_run(cmd, **kwargs):
            return subprocess.CompletedProcess(cmd, 44, stdout="")

        monkeypatch.setattr(subprocess, "run", fake_run)

        assert _keychain_token() is None

    def test_returns_none_when_security_unavailable(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "darwin")

        def fake_run(cmd, **kwargs):
            raise FileNotFoundError("security")

        monkeypatch.setattr(subprocess, "run", fake_run)

        assert _keychain_token() is None


class TestCredentialsFileToken:
    def test_json_access_token(self, tmp_path):
        path = tmp_path / "credentials"
        path.write_text(json.dumps({"access_token": "json-tok"}))

        assert _credentials_file_token(path) == "json-tok"

    def test_json_token_key(self, tmp_path):
        path = tmp_path / "credentials"
        path.write_text(json.dumps({"token": "json-tok-2"}))

        assert _credentials_file_token(path) == "json-tok-2"

    def test_raw_token(self, tmp_path):
        path = tmp_path / "credentials"
        path.write_text("raw-token\n")

        assert _credentials_file_token(path) == "raw-token"

    def test_absent_file(self, tmp_path):
        assert _credentials_file_token(tmp_path / "nope") is None

    def test_empty_file(self, tmp_path):
        path = tmp_path / "credentials"
        path.write_text("\n")

        assert _credentials_file_token(path) is None


class TestEnvironmentSnapshot:
    def test_snapshot_contents(self):
        import abtem

        snapshot = environment_snapshot()

        assert snapshot["python"] == sys.version.split()[0]
        assert isinstance(snapshot["platform"], str) and snapshot["platform"]
        assert snapshot["packages"]["abTEM"] == abtem.__version__
        for package in ("numpy", "dask", "ase", "zarr"):
            assert isinstance(snapshot["packages"][package], str)
        # timezone-aware ISO 8601 timestamp
        assert "T" in snapshot["timestamp"]
        assert snapshot["timestamp"].endswith("+00:00") or snapshot[
            "timestamp"
        ].endswith("Z")

    def test_snapshot_is_json_serializable(self):
        json.dumps(environment_snapshot())

    def test_snapshot_tolerates_missing_package(self):
        snapshot = environment_snapshot(packages=("numpy", "not-a-real-package"))

        assert snapshot["packages"]["not-a-real-package"] is None
        assert isinstance(snapshot["packages"]["numpy"], str)
