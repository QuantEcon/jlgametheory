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

copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True

from importlib.metadata import PackageNotFoundError, version as _version
try:
    version = release = _version("jlgametheory")
except PackageNotFoundError:
    version = release = ""


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
    app.connect("doctree-resolved", _strip_autosummary_anchors)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
