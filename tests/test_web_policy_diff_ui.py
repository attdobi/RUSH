from __future__ import annotations

from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]


def test_policy_diff_ui_collapses_parse_errors_and_loads_them() -> None:
    source = (_REPO_ROOT / "web" / "policy-diff.js").read_text(encoding="utf-8")

    assert "/api/policy/proposals?include_errors=true" in source
    assert "<details class=\"proposal-status-group proposal-parse-errors\">" in source
    assert "Parse errors (${parseErrors.length})" in source
    assert "proposalStatusGroups(reviewable)" in source
    assert "proposal-card-${statusKey}" in source
    assert "proposalVersionText(proposal)" in source


def test_policy_diff_version_chips_have_single_current_placeholder() -> None:
    source = (_REPO_ROOT / "web" / "policy-diff.js").read_text(encoding="utf-8")
    html = (_REPO_ROOT / "web" / "index.html").read_text(encoding="utf-8")

    assert "Policy ${currentVersion} · current" in source
    assert "payload.build_version || (payload.base_version ? `${payload.base_version}+`" in source
    assert "proposalVersionArrow" in html
    assert "proposalBuildVersionLabel" in html
    # Cache-buster: any versioned query string on policy-diff.js is fine.
    import re as _re
    assert _re.search(r"policy-diff\.js\?v=[^\"']+", html)
