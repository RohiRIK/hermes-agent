"""Phase 4 runtime extension: optional ``provider`` on ``SubagentLaunchRequest``.

Contract (audit/runtime-audit.md Section 4, bob-runtime-verification.md):
- ``provider`` is appended to the dataclass without shifting any existing
  positional field's meaning -- old positional callers keep working.
- ``provider`` set without ``model`` is rejected before any child spawns.
- ``provider`` set (with ``model``) resolves the SAME full credential/endpoint
  bundle ``delegate_task`` uses (never a bare ``override_provider=``).
- ``provider`` omitted preserves the exact historical behavior: no override_*
  kwargs at all, full parent inheritance (never silently switches on
  config-level ``delegation.provider``).
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent.subagent_lifecycle import (
    SubagentLaunchRequest,
    SubagentLifecycleError,
    SubagentLifecycleService,
)


@pytest.fixture
def parent():
    return SimpleNamespace(session_id="parent-1", enabled_toolsets=["file"])


# ── positional compatibility ─────────────────────────────────────────────────


def test_provider_field_is_appended_last_positional_calls_unaffected():
    """Every pre-existing field must still bind by the same position; the new
    field only extends the tuple, it never inserts before an old field."""
    req = SubagentLaunchRequest(
        "goal text", "context text", "orchestrator", "some-model",
        ("file",), ("dangerous_tool",), None, "sess-1", "corr-1", {"k": "v"}, None,
    )
    assert req.goal == "goal text"
    assert req.context == "context text"
    assert req.role == "orchestrator"
    assert req.model == "some-model"
    assert req.allowed_toolsets == ("file",)
    assert req.blocked_tools == ("dangerous_tool",)
    assert req.working_directory is None
    assert req.parent_session_id == "sess-1"
    assert req.correlation_id == "corr-1"
    assert req.metadata == {"k": "v"}
    assert req.timeout_seconds is None
    assert req.provider is None  # new field defaults, doesn't shift anything


def test_provider_defaults_to_none_for_keyword_construction():
    req = SubagentLaunchRequest(goal="x")
    assert req.provider is None


# ── provider requires model ──────────────────────────────────────────────────


def test_provider_without_model_rejected_before_launch(monkeypatch, parent):
    build = MagicMock()
    monkeypatch.setattr("tools.delegate_tool._build_child_preserving_parent_tools", build)
    service = SubagentLifecycleService(lambda: parent)
    with pytest.raises(SubagentLifecycleError, match="provider requires model"):
        service.launch(SubagentLaunchRequest(goal="x", provider="some-provider", model=None))
    build.assert_not_called()


def test_malformed_provider_rejected(monkeypatch, parent):
    build = MagicMock()
    monkeypatch.setattr("tools.delegate_tool._build_child_preserving_parent_tools", build)
    service = SubagentLifecycleService(lambda: parent)
    for bad in ("", "  padded  ", 12345):
        with pytest.raises(SubagentLifecycleError):
            service.launch(SubagentLaunchRequest(goal="x", provider=bad, model="m"))
    build.assert_not_called()


# ── provider set: full-bundle wiring, never a bare pass-through ────────────


def test_provider_and_model_resolves_full_bundle(monkeypatch, parent):
    captured = {}

    def fake_resolve(cfg, _parent):
        assert cfg == {"model": "b-model", "provider": "provider-b"}
        return {
            "model": "b-model", "provider": "custom", "base_url": "https://provider-b.example/v1",
            "api_key": "provider-b-key", "api_mode": "chat_completions",
            "request_overrides": {"extra_body": {"thinking": {"type": "disabled"}}},
            "max_output_tokens": 4096, "command": None, "args": [],
        }

    def fake_build(**kwargs):
        captured.update(kwargs)
        return MagicMock(_subagent_id="sa-1", provider="custom", model="b-model", _delegate_role="leaf", _delegate_depth=1)

    monkeypatch.setattr("tools.delegate_tool_config._resolve_delegation_credentials", fake_resolve)
    monkeypatch.setattr("tools.delegate_tool._build_child_preserving_parent_tools", fake_build)
    service = SubagentLifecycleService(lambda: parent)
    handle = service.launch(SubagentLaunchRequest(goal="x", provider="provider-b", model="b-model"))

    assert captured["model"] == "b-model"
    assert captured["override_provider"] == "custom"
    # Full bundle, not a bare override_provider= pass-through (gap F).
    assert captured["override_base_url"] == "https://provider-b.example/v1"
    assert captured["override_api_key"] == "provider-b-key"
    assert captured["override_api_mode"] == "chat_completions"
    assert handle.provider == "custom"
    assert handle.model == "b-model"


def test_provider_resolution_failure_raises_lifecycle_error(monkeypatch, parent):
    def fake_resolve(cfg, _parent):
        raise ValueError("Cannot resolve delegation provider 'broken'.")

    build = MagicMock()
    monkeypatch.setattr("tools.delegate_tool_config._resolve_delegation_credentials", fake_resolve)
    monkeypatch.setattr("tools.delegate_tool._build_child_preserving_parent_tools", build)
    service = SubagentLifecycleService(lambda: parent)
    with pytest.raises(SubagentLifecycleError):
        service.launch(SubagentLaunchRequest(goal="x", provider="broken", model="m"))
    build.assert_not_called()


# ── provider omitted: exact historical behavior preserved ───────────────────


def test_provider_omitted_passes_no_override_kwargs(monkeypatch, parent):
    """The historical contract: with no provider, launch() must call the
    child builder with ONLY the pre-existing kwargs -- no override_* keys at
    all -- so a child fully inherits the parent's route exactly as before
    this feature existed (never silently switches on config-level
    delegation.provider)."""
    captured = {}

    def fake_build(**kwargs):
        captured.update(kwargs)
        return MagicMock(_subagent_id="sa-1", provider="inherited", model="inherited-model", _delegate_role="leaf", _delegate_depth=1)

    monkeypatch.setattr("tools.delegate_tool._build_child_preserving_parent_tools", fake_build)
    service = SubagentLifecycleService(lambda: parent)
    service.launch(SubagentLaunchRequest(goal="x", model="requested-model"))

    override_keys = [k for k in captured if k.startswith("override_")]
    assert override_keys == []
    assert captured["model"] == "requested-model"
