from __future__ import annotations

from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]


def test_policy_diff_ui_collapses_parse_errors_and_loads_them() -> None:
    source = (_REPO_ROOT / "web" / "policy-diff.js").read_text(encoding="utf-8")

    assert "/api/policy/proposals?include_errors=true" in source
    assert "activePolicyArea()" in source
    assert "area," in source
    assert "selectedPolicyVersion()" in source
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
    # The proposal UI was unmounted with the Inspect tab (2026-07-07); the
    # module and its markup contract survive for a future re-mount, but
    # index.html no longer carries the host elements.
    assert "proposalVersionArrow" not in html
    assert "proposalBuildVersionLabel" not in html
    # The module is no longer loaded by the page at all.
    assert "policy-diff.js" not in html
