"""Add investigation controls to the native lab; preserve the original About."""
from __future__ import annotations


def enhance_lab_html(html: str) -> str:
    if 'id="studioView"' in html:
        return html.replace('src="about.js', 'src="studio-about.js')
    if 'id="experiment"' not in html or 'src="lab-evidence.js' in html:
        return html
    assets = '''
  <link rel="stylesheet" href="lab-evidence.css" />
  <script src="lab-evidence-core.js"></script>
  <script src="lab-evidence.js"></script>
  <script src="research-addendum.js"></script>
'''
    return html.replace('</body>', assets + '</body>', 1)
