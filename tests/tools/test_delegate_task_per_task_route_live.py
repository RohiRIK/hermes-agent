"""Phase 4 runtime extension -- REAL end-to-end route-propagation proof.

SYNTHETIC PROVIDER TEST: both "providers" here are local HTTP servers on
127.0.0.1 with fabricated API keys, configured as named ``custom_providers``
entries in a REAL temporary HERMES_HOME config.yaml. This exercises the real
resolver chain end to end -- ``delegate_task`` schema/validation ->
``_resolve_task_credentials`` -> ``_resolve_delegation_credentials`` ->
``hermes_cli.runtime_provider.resolve_runtime_provider`` -> real config load
-> real ``AIAgent`` child construction -- and proves route/credential
PROPAGATION with one real HTTP request per child. It does NOT prove anything
about any real upstream provider's health, and no internet access is made
(both endpoints are 127.0.0.1). ``_run_single_child`` (the full multi-turn
conversation loop) is replaced with a single real wire call per child so the
test exercises exactly the new routing code, not unrelated conversation-loop
machinery.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import MagicMock, patch

import pytest

import hermes_cli.config as hc
from tools.delegate_tool import delegate_task


class _RecordingHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_a):
        pass

    def do_POST(self):
        n = int(self.headers.get("content-length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        self.server.owner.requests.append({
            "path": self.path, "auth": self.headers.get("authorization"), "body": body,
        })
        if not self.path.endswith("/chat/completions"):
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        chunk = {
            "id": "c1", "object": "chat.completion.chunk", "created": 1, "model": body.get("model"),
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": "done"}, "finish_reason": "stop"}],
        }
        self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


class _FakeProviderServer:
    def __init__(self):
        self.requests: list[dict] = []
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _RecordingHandler)
        self.server.owner = self
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def two_fake_providers(tmp_path, monkeypatch):
    server_a, server_b = _FakeProviderServer(), _FakeProviderServer()
    home = tmp_path / "hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        "model:\n  default: fallback-model\n"
        "custom_providers:\n"
        f"  - name: provider-a\n    base_url: {server_a.base_url}\n    api_key: synthetic-key-a\n"
        f"  - name: provider-b\n    base_url: {server_b.base_url}\n    api_key: synthetic-key-b\n"
        "  - name: broken-default\n    base_url: http://127.0.0.1:1\n    api_key: unused-key\n"
        "delegation:\n  max_iterations: 1\n"
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    hc._LOAD_CONFIG_CACHE.clear()
    try:
        yield server_a, server_b
    finally:
        hc._LOAD_CONFIG_CACHE.clear()
        server_a.close()
        server_b.close()


def _live_parent(**extra):
    """Real-enough parent for genuine ``_build_child_agent`` construction
    (not just a mocked-out stand-in) -- concrete values for every attribute
    the real construction/resolution path reads."""
    parent = MagicMock()
    parent.base_url = "https://openrouter.ai/api/v1"
    parent.api_key = "parent-key-never-used-by-explicit-routes"
    parent.provider = "openrouter"
    parent.api_mode = "chat_completions"
    parent.model = "parent/model-never-used-by-explicit-routes"
    parent.platform = "cli"
    parent.providers_allowed = None
    parent.providers_ignored = None
    parent.providers_order = None
    parent.provider_sort = None
    parent.openrouter_min_coding_score = None
    parent.request_overrides = None
    parent.max_tokens = None
    parent.acp_command = None
    parent.acp_args = []
    parent.capabilities = None
    parent.enabled_toolsets = []
    parent.valid_tool_names = set()
    parent.prefill_messages = None
    parent.session_id = "live-parent-session"
    parent._session_db = None
    parent._delegate_depth = 0
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    parent._print_fn = None
    parent._client_kwargs = {}
    parent.client = None
    parent._current_turn_id = ""
    parent._subagent_id = None
    parent.tool_progress_callback = None
    parent.thinking_callback = None
    for k, v in extra.items():
        setattr(parent, k, v)
    return parent


def _wire_single_child(task_index, goal, child=None, parent_agent=None, **_kwargs):
    """Stand-in for ``_run_single_child``: one REAL wire call through the
    child's own resolved route, instead of the full multi-turn conversation
    loop (unrelated machinery for this feature)."""
    from agent.chat_completion_helpers import interruptible_streaming_api_call

    response = interruptible_streaming_api_call(
        child, {"model": child.model, "messages": [{"role": "user", "content": goal}]},
    )
    entry = {
        "task_index": task_index, "status": "completed",
        "summary": response.choices[0].message.content, "api_calls": 1, "duration_seconds": 0.01,
        "model": getattr(child, "model", None), "provider": getattr(child, "provider", None),
        "_child_role": getattr(child, "_delegate_role", None),
    }
    requested_model = getattr(child, "_requested_model", None)
    requested_provider = getattr(child, "_requested_provider", None)
    if requested_model is not None:
        entry["requested_model"] = requested_model
    if requested_provider is not None:
        entry["requested_provider"] = requested_provider
    return entry


def test_two_explicit_providers_route_to_distinct_endpoints_no_cross_contamination(two_fake_providers):
    server_a, server_b = two_fake_providers
    with patch("tools.delegate_tool._run_single_child", side_effect=_wire_single_child):
        result = json.loads(delegate_task(
            tasks=[
                {"goal": "Task routed to provider A", "model": "model-a", "provider": "provider-a"},
                {"goal": "Task routed to provider B", "model": "model-b", "provider": "provider-b"},
            ],
            parent_agent=_live_parent(),
            background=False,
        ))

    assert "error" not in result, result
    assert [r["status"] for r in result["results"]] == ["completed", "completed"]

    a_completions = [r for r in server_a.requests if r["path"].endswith("/chat/completions")]
    b_completions = [r for r in server_b.requests if r["path"].endswith("/chat/completions")]
    assert len(a_completions) == 1, "expected exactly one wire call to reach provider A"
    assert len(b_completions) == 1, "expected exactly one wire call to reach provider B"

    assert a_completions[0]["auth"] == "Bearer synthetic-key-a"
    assert a_completions[0]["body"]["model"] == "model-a"
    assert b_completions[0]["auth"] == "Bearer synthetic-key-b"
    assert b_completions[0]["body"]["model"] == "model-b"

    # No cross-contamination: neither server ever saw the other's key/host.
    assert all(r["auth"] != "Bearer synthetic-key-b" for r in server_a.requests)
    assert all(r["auth"] != "Bearer synthetic-key-a" for r in server_b.requests)

    # Per-child resolved provider/model surfaced on the result entry, not a
    # batch-uniform value (heterogeneous batch correctness).
    entries = {e["task_index"]: e for e in result["results"]}
    assert entries[0]["model"] == "model-a"
    assert entries[1]["model"] == "model-b"
    assert entries[0].get("provider") == "provider-a"
    assert entries[1].get("provider") == "provider-b"
    assert entries[0].get("requested_model") == "model-a"
    assert entries[0].get("requested_provider") == "provider-a"


def test_all_explicit_batch_survives_broken_default_provider_real_config(two_fake_providers):
    """delegation.provider globally pinned (via credentials_cfg) to a
    provider that cannot actually connect must not block a batch where every
    task pins its own working provider+model."""
    server_a, server_b = two_fake_providers
    with patch("tools.delegate_tool._run_single_child", side_effect=_wire_single_child):
        result = json.loads(delegate_task(
            tasks=[
                {"goal": "Task routed to provider A", "model": "model-a", "provider": "provider-a"},
                {"goal": "Task routed to provider B", "model": "model-b", "provider": "provider-b"},
            ],
            parent_agent=_live_parent(),
            background=False,
            credentials_cfg={"provider": "broken-default"},
        ))
    assert "error" not in result, result
    assert [r["status"] for r in result["results"]] == ["completed", "completed"]
    assert len([r for r in server_a.requests if r["path"].endswith("/chat/completions")]) == 1
    assert len([r for r in server_b.requests if r["path"].endswith("/chat/completions")]) == 1


def test_model_only_override_inherits_delegated_provider_real_config(two_fake_providers):
    """model-only per-task override (no provider) uses the effective
    DELEGATED provider (credentials_cfg pins provider-a for the whole
    batch) -- never parses the model string for a provider hint."""
    server_a, server_b = two_fake_providers
    with patch("tools.delegate_tool._run_single_child", side_effect=_wire_single_child):
        result = json.loads(delegate_task(
            tasks=[{"goal": "Model-only override task", "model": "a-alternate-model"}],
            parent_agent=_live_parent(),
            background=False,
            credentials_cfg={"provider": "provider-a"},
        ))
    assert "error" not in result, result
    a_completions = [r for r in server_a.requests if r["path"].endswith("/chat/completions")]
    assert len(a_completions) == 1
    assert a_completions[0]["auth"] == "Bearer synthetic-key-a"
    assert a_completions[0]["body"]["model"] == "a-alternate-model"
    assert not server_b.requests
