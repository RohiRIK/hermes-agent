#!/usr/bin/env python3
"""Per-task ``model``/``provider`` overrides on ``delegate_task(tasks=[...])``.

Phase 4 runtime extension (dynamic-model-selection plan): each task in a batch
may pin its own ``model``/``provider`` instead of the whole batch sharing one
resolved credential bundle. Contract (see
audit/runtime-audit.md Section 3 and the approved plan):

- No per-task ``model``/``provider`` -> byte-identical to today (batch/parent bundle).
- ``model`` only -> batch/parent bundle with just the model replaced.
- ``provider`` (+ required ``model``) -> a FRESH full-bundle resolution
  (``_resolve_delegation_credentials``), never a bare ``override_provider=`` pass-through.
- ``provider`` without ``model`` on ANY task rejects the WHOLE batch before any
  child spawns.
- Malformed identifiers (non-string, empty, or surrounding whitespace) are
  rejected outright, never repaired/stripped.
- A batch where every task supplies its own explicit ``provider``+``model``
  must not fail just because the unused batch/default provider is broken.
- A later task's construction failure cleans up already-built earlier children.
"""

import json
import threading
import unittest
from unittest.mock import MagicMock, patch

from tools.delegate_tool import delegate_task


def _make_mock_parent(depth=0, **extra):
    parent = MagicMock()
    parent.base_url = "https://openrouter.ai/api/v1"
    parent.api_key = "test-key"
    parent.provider = "openrouter"
    parent.api_mode = "chat_completions"
    parent.model = "anthropic/claude-sonnet-4"
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
    parent._session_db = None
    parent._delegate_depth = depth
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    parent._print_fn = None
    parent._client_kwargs = {}
    parent.client = None
    parent.tool_progress_callback = None
    parent.thinking_callback = None
    for k, v in extra.items():
        setattr(parent, k, v)
    return parent


def _call(tasks, parent=None, **kw):
    return json.loads(delegate_task(tasks=tasks, parent_agent=parent or _make_mock_parent(), **kw))


GOOD_A = "Refactor the login handler to use the new session helper"
GOOD_B = "Write regression tests for the session expiry watcher"


def _completed(idx):
    return {
        "task_index": idx, "status": "completed", "summary": "ok",
        "api_calls": 1, "duration_seconds": 1.0, "_child_role": None,
    }


# ── No override: byte-identical to today ───────────────────────────────────


class TestNoOverrideUnaffected(unittest.TestCase):
    def test_batch_with_no_model_or_provider_still_runs(self):
        with patch("tools.delegate_tool._run_single_child") as mock_run:
            mock_run.side_effect = [_completed(0), _completed(1)]
            result = _call([{"goal": GOOD_A}, {"goal": GOOD_B}])
        self.assertNotIn("error", result)
        self.assertEqual(len(result["results"]), 2)


# ── model-only override ─────────────────────────────────────────────────────


class TestModelOnlyOverride(unittest.TestCase):
    def test_model_only_inherits_batch_provider_and_credentials(self):
        captured = {}
        real_build = None
        import tools.delegate_tool as dt

        real_build = dt._build_child_preserving_parent_tools

        def spy(**kwargs):
            captured.setdefault("calls", []).append(kwargs)
            return MagicMock(_subagent_id="sa-x")

        with patch("tools.delegate_tool._build_child_preserving_parent_tools", side_effect=spy), \
             patch("tools.delegate_tool._run_batch") as mock_run_batch:
            mock_run_batch.return_value = "{}"
            delegate_task(
                tasks=[{"goal": GOOD_A, "model": "openrouter/some-other-model"}],
                parent_agent=_make_mock_parent(),
            )
        calls = captured["calls"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["model"], "openrouter/some-other-model")
        # No provider override on the task -> the child still inherits the
        # parent/batch route (override_provider stays None here since no
        # delegation.provider/base_url is configured for this mock parent).
        self.assertIsNone(calls[0]["override_provider"])


# ── provider (+model) override: fresh full bundle, never a bare pass-through ──


class TestProviderPlusModelOverride(unittest.TestCase):
    @patch("hermes_cli.runtime_provider.resolve_runtime_provider")
    def test_provider_and_model_resolves_full_bundle_not_bare_override(self, mock_resolve):
        mock_resolve.return_value = {
            "provider": "custom",
            "base_url": "https://provider-b.example/v1",
            "api_key": "provider-b-key",
            "api_mode": "chat_completions",
            "request_overrides": {"extra_body": {"thinking": {"type": "disabled"}}},
            "max_output_tokens": 4096,
        }
        captured = {}

        def spy(**kwargs):
            captured.setdefault("calls", []).append(kwargs)
            return MagicMock(_subagent_id="sa-x")

        with patch("tools.delegate_tool._build_child_preserving_parent_tools", side_effect=spy), \
             patch("tools.delegate_tool._run_batch") as mock_run_batch:
            mock_run_batch.return_value = "{}"
            delegate_task(
                tasks=[{"goal": GOOD_A, "model": "b-model", "provider": "provider-b"}],
                parent_agent=_make_mock_parent(),
            )
        call = captured["calls"][0]
        self.assertEqual(call["model"], "b-model")
        # _resolve_delegation_credentials substitutes back the originally-requested name when
        # the runtime resolves to bare "custom" (matches its existing, unchanged behavior).
        self.assertEqual(call["override_provider"], "provider-b")
        # The FULL bundle must be present -- never just override_provider with
        # everything else left to inherit the parent (gap F in the audit).
        self.assertEqual(call["override_base_url"], "https://provider-b.example/v1")
        self.assertEqual(call["override_api_key"], "provider-b-key")
        self.assertEqual(call["override_api_mode"], "chat_completions")


# ── provider without model: reject the whole batch, zero spawns ────────────


class TestProviderRequiresModel(unittest.TestCase):
    def test_provider_without_model_rejects_whole_batch(self):
        with patch("tools.delegate_tool._build_child_preserving_parent_tools") as mock_build, \
             patch("tools.delegate_tool._run_single_child") as mock_run:
            result = _call([
                {"goal": GOOD_A, "provider": "some-provider"},
                {"goal": GOOD_B},
            ])
        self.assertIn("error", result)
        self.assertIn("model", result["error"].lower())
        mock_build.assert_not_called()
        mock_run.assert_not_called()

    def test_second_task_bad_provider_prevents_first_task_spawn(self):
        """A bad task later in the batch must not let an earlier good task spawn."""
        with patch("tools.delegate_tool._build_child_preserving_parent_tools") as mock_build, \
             patch("tools.delegate_tool._run_single_child") as mock_run:
            result = _call([
                {"goal": GOOD_A},
                {"goal": GOOD_B, "provider": "orphan-provider"},
            ])
        self.assertIn("error", result)
        mock_build.assert_not_called()
        mock_run.assert_not_called()


# ── malformed identifiers: reject, never repair ─────────────────────────────


class TestMalformedIdentifiersRejected(unittest.TestCase):
    def test_whitespace_padded_model_rejected(self):
        result = _call([{"goal": GOOD_A, "model": "  gpt-4  "}])
        self.assertIn("error", result)

    def test_empty_string_model_rejected(self):
        result = _call([{"goal": GOOD_A, "model": ""}])
        self.assertIn("error", result)

    def test_empty_string_provider_rejected(self):
        result = _call([{"goal": GOOD_A, "model": "m", "provider": ""}])
        self.assertIn("error", result)

    def test_non_string_model_rejected(self):
        result = _call([{"goal": GOOD_A, "model": 12345}])
        self.assertIn("error", result)

    def test_non_string_provider_rejected(self):
        result = _call([{"goal": GOOD_A, "model": "m", "provider": ["p"]}])
        self.assertIn("error", result)

    def test_whitespace_padded_provider_rejected(self):
        result = _call([{"goal": GOOD_A, "model": "m", "provider": " provider-b "}])
        self.assertIn("error", result)

    def test_no_spawn_on_invalid_task_in_multi_task_batch(self):
        with patch("tools.delegate_tool._build_child_preserving_parent_tools") as mock_build:
            result = _call([{"goal": GOOD_A}, {"goal": GOOD_B, "model": "  bad  "}])
        self.assertIn("error", result)
        mock_build.assert_not_called()


# ── all-explicit batch unaffected by a broken/unavailable default ──────────


class TestAllExplicitBatchIgnoresBrokenDefault(unittest.TestCase):
    @patch("hermes_cli.runtime_provider.resolve_runtime_provider")
    def test_all_tasks_explicit_survive_broken_delegation_provider_config(self, mock_resolve):
        """delegation.provider globally configured to something broken must
        not block a batch where every task pins its own provider+model."""

        def side_effect(*, requested, target_model=None):
            if requested == "totally-broken-default-provider":
                raise RuntimeError("simulated: default provider is unavailable")
            return {
                "provider": "custom",
                "base_url": f"https://{requested}.example/v1",
                "api_key": f"{requested}-key",
                "api_mode": "chat_completions",
                "request_overrides": None,
                "max_output_tokens": None,
            }

        mock_resolve.side_effect = side_effect
        captured = {}

        def spy(**kwargs):
            captured.setdefault("calls", []).append(kwargs)
            return MagicMock(_subagent_id=f"sa-{len(captured.get('calls', []))}")

        cfg = {"provider": "totally-broken-default-provider"}
        with patch("tools.delegate_tool._load_config", return_value=cfg), \
             patch("tools.delegate_tool._build_child_preserving_parent_tools", side_effect=spy), \
             patch("tools.delegate_tool._run_batch") as mock_run_batch:
            mock_run_batch.return_value = "{}"
            result_json = delegate_task(
                tasks=[
                    {"goal": GOOD_A, "model": "model-a", "provider": "provider-a"},
                    {"goal": GOOD_B, "model": "model-b", "provider": "provider-b"},
                ],
                parent_agent=_make_mock_parent(),
            )
        self.assertEqual(result_json, "{}")
        calls = captured["calls"]
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["override_base_url"], "https://provider-a.example/v1")
        self.assertEqual(calls[1]["override_base_url"], "https://provider-b.example/v1")

    def test_mixed_batch_still_needs_default_for_inheriting_task(self):
        """One explicit task + one inheriting task -> the broken default DOES
        block the whole call, because the inheriting task actually needs it."""
        cfg = {"provider": "totally-broken-default-provider"}
        with patch("tools.delegate_tool._load_config", return_value=cfg), \
             patch(
                 "hermes_cli.runtime_provider.resolve_runtime_provider",
                 side_effect=RuntimeError("simulated: default provider is unavailable"),
             ):
            result = _call([
                {"goal": GOOD_A, "model": "model-a", "provider": "provider-a"},
                {"goal": GOOD_B},
            ])
        self.assertIn("error", result)


# ── cleanup on partial batch construction failure ───────────────────────────


class TestPartialBatchCleanup(unittest.TestCase):
    @patch("hermes_cli.runtime_provider.resolve_runtime_provider")
    def test_later_task_failure_cleans_up_earlier_built_children(self, mock_resolve):
        mock_resolve.return_value = {
            "provider": "custom", "base_url": "https://provider-a.example/v1",
            "api_key": "provider-a-key", "api_mode": "chat_completions",
            "request_overrides": None, "max_output_tokens": None,
        }
        built = []

        def spy(*, task_index, **kwargs):
            if task_index == 1:
                raise ValueError("Delegation provider 'broken' resolved but has no API key.")
            child = MagicMock(_subagent_id=f"sa-{task_index}")
            built.append(child)
            return child

        closed = []
        detached = []
        with patch("tools.delegate_tool._build_child_preserving_parent_tools", side_effect=spy), \
             patch("tools.delegate_tool._close_child", side_effect=lambda c, *_a, **_k: closed.append(c)), \
             patch("tools.delegate_tool._detach_child", side_effect=lambda p, c: detached.append(c)):
            result = _call([
                {"goal": GOOD_A, "model": "model-a", "provider": "provider-a"},
                {"goal": GOOD_B, "model": "model-b", "provider": "broken"},
            ])
        self.assertIn("error", result)
        self.assertEqual(len(built), 1)
        self.assertIn(built[0], closed)
        self.assertIn(built[0], detached)


if __name__ == "__main__":
    unittest.main()
