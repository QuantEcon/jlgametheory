import os, sys
sys.path.insert(0, os.path.abspath('..'))

project = "jlgametheory"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
]
autosummary_generate = True
autodoc_mock_imports = ["juliacall"]
add_module_names = False
html_theme = "sphinxdoc"

templates_path = ["_templates"]
autodoc_typehints = "none"
add_function_parentheses = False

default_role = "code"

html_show_sourcelink = False

html_static_path = ["_static"]
html_js_files = ["autosummary_fixes.js", "autosummary_no_underline.css"]

copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True

try:
    import jlgametheory as _pkg
    version = release = getattr(_pkg, "__version__", "")
except Exception:
    version = release = ""
