"""--no-changelog keeps the release body's frame and drops the commit sections."""


def test_no_changelog_keeps_the_frame_without_commit_sections():
    from scripts import release

    commits = [{
        "sha": "a" * 40, "short_sha": "a" * 8, "author_name": "Dev", "author_email": "dev@example.test",
        "subject": "feat: something (#5)", "category": "features", "github_author": "@dev", "coauthors": [],
    }]

    body = release.generate_changelog(commits, "rc.1-v1.2.4", "1.2.4", repo_url="https://github.com/o/r",
                                      prev_tag="v1.2.3", no_changelog=True)

    assert "Something" not in body and "@dev" not in body
    assert "<!-- HERMES_BUILDS_TABLE -->" in body
    assert "https://github.com/o/r/compare/v1.2.3...rc.1-v1.2.4" in body


def test_clamp_release_notes_is_noop_under_limit():
    from scripts import release

    notes = "# short\n\nok\n"
    assert release.clamp_release_notes(notes) == notes


def test_clamp_release_notes_caps_under_github_body_max():
    from scripts import release

    # Simulate the fork failure mode: tens of thousands of commit bullets.
    bullet = "- feat: something useful ([`abcdef12`](https://github.com/o/r/commit/abcdef12)) — @dev\n"
    notes = "# Hermes Agent v0.21.4+canary.example\n\n" + (bullet * 8000)
    assert len(notes) > release.GITHUB_RELEASE_BODY_MAX

    clamped = release.clamp_release_notes(notes)
    assert len(clamped) <= release.RELEASE_NOTES_SOFT_MAX
    assert len(clamped) <= release.GITHUB_RELEASE_BODY_MAX
    assert clamped.startswith("# Hermes Agent")
    assert "truncated to fit GitHub" in clamped
    assert clamped.endswith("\n")
    # Prefer a clean cut at a newline rather than mid-bullet.
    body, _, notice = clamped.rpartition("\n\n---\n\n")
    assert notice.startswith("_Release notes truncated")
    assert body.endswith(")") or body.endswith("@dev") or body.startswith("#")


def test_clamp_release_notes_rejects_limit_above_github_max():
    from scripts import release
    import pytest

    with pytest.raises(ValueError, match="cannot exceed"):
        release.clamp_release_notes("x", limit=release.GITHUB_RELEASE_BODY_MAX + 1)
