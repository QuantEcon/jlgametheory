import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = "jlgametheory"
copyright = "2025, QuantEcon"
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
