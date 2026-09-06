"""Progressively enhance the original research lab; leave its DOM contracts intact."""
from __future__ import annotations

import re

ASSETS = '''\n  <link rel="stylesheet" href="research.css" />
  <script defer src="research-core.js"></script>
  <script defer src="research.js"></script>\n'''


def enhance_lab_html(html: str) -> str:
    if 'id="studioView"' in html:
        # Keep the pre-existing shadow sandbox's methods module isolated.
        return html.replace('src="about.js', 'src="studio-about.js')
    if 'id="experiment"' not in html or re.search(r'''src=["']research\.js(?:[?"'])''', html):
        return html
    html = re.sub(r"<title>.*?</title>", "<title>RUSH — Policy Learning Research Lab</title>", html, count=1)
    # Correct the original help text without changing any control ids or values.
    html = html.replace("No expensive model ever scores quality.", "Deterministic scorers compare panel predictions with reference labels.")
    html = html.replace("because no\n            policy edit can fix a judge that isn't reading the policy.", "this is a diagnostic heuristic, not proof that a judge ignores the policy.")
    html = html.replace("the\n            benchmark readout adds two passes over the fixed 1,000-image validation split.", "the benchmark readout adds passes over the configured validation manifest.")
    html = html.replace("<label>Test size T", "<label>Validation size T")
    return html.replace("</head>", ASSETS + "</head>", 1)
