import os
import sys
from datetime import datetime
sys.path.insert(0, os.path.abspath('..'))

project = "jlgametheory"
year = datetime.now().year
copyright = f"2025-{year}, QuantEcon"
author = "QuantEcon"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
]
autosummary_generate = True
autodoc_mock_imports = ["juliacall"]
html_theme = "sphinxdoc"

templates_path = ["_templates"]
autodoc_typehints = "none"
add_function_parentheses = False

default_role = "code"

html_show_sourcelink = False

html_static_path = ["_static"]
html_css_files = ["custom.css"]

# Documentation channel ("stable" or "latest") for the channel switcher
# in the navigation bar, set by the docs deployment workflow. Empty for
# local builds, where both channels render as links.
docs_channel = os.environ.get("JLGAMETHEORY_DOCS_CHANNEL", "")
docs_base_url = "https://quantecon.github.io/jlgametheory"

html_context = {
    "docs_channel": docs_channel,
    "docs_channels": [
        {
            "name": "stable",
            "label": "stable",
            "description": "Stable release documentation",
            "base_url": f"{docs_base_url}/stable/",
        },
        {
            "name": "latest",
            "label": "latest",
            "description": "Latest development documentation",
            "base_url": f"{docs_base_url}/latest/",
        },
    ],
}

# Canonical URL, pointing at the stable channel from both channels
html_baseurl = f"{docs_base_url}/stable/"

copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True

from importlib.metadata import PackageNotFoundError, version as _version
try:
    version = release = _version("jlgametheory")
except PackageNotFoundError:
    version = release = ""


def _strip_genindex_module_annotation(app, pagename, templatename, context,
                                      doctree):
    """Drop the " (in module ...)" annotation from general index entries
    (e.g. show ``lrsnash()`` instead of ``lrsnash() (in module
    jlgametheory)``).
    """
    if pagename != "genindex":
        return
    import re

    def strip(name):
        return re.sub(r"\s*\(in module .*?\)", "", name)

    context["genindexentries"] = [
        (letter, [(strip(name), rest) for name, rest in entries])
        for letter, entries in context["genindexentries"]
    ]


def _strip_autosummary_anchors(app, doctree, docname):
    """Point autosummary table links at the page, not the object's anchor.

    Sphinx links each autosummary entry to the object's anchor
    (e.g. ``_autosummary/jlgametheory.lrsnash.html#jlgametheory.lrsnash``);
    drop the ``#fragment`` so the link targets the page itself.
    """
    from docutils import nodes

    for table in doctree.findall(nodes.table):
        if "autosummary" not in table.get("classes", []):
            continue
        for ref in table.findall(nodes.reference):
            uri = ref.get("refuri")
            if uri and "#" in uri:
                ref["refuri"] = uri.split("#", 1)[0]


def setup(app):
    app.connect("html-page-context", _strip_genindex_module_annotation)
    app.connect("doctree-resolved", _strip_autosummary_anchors)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
