# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import wsinfer

project = "WSInfer"
copyright = "2023, Jakub Kaczmarzyk"
author = "Jakub Kaczmarzyk"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.intersphinx",  # for links
    "sphinx.ext.napoleon",  # for google style docstrings
    "sphinx.ext.viewcode",  # add links to code
    "autoapi.extension",  # to document the wsinfer api
    "sphinx_click",  # to document click command line
    "sphinx_copybutton",  # add copy button to top-right of code blocks
]

# Internationalization.
language = "en"

# AutoAPI options.
autoapi_type = "python"
autoapi_dirs = ["../wsinfer"]
autoapi_options = [
    "members",
    "undoc-members",
    # "private-members",
    "show-inheritance",
    "show-module-summary",
    "special-members",
    "imported-members",
]
autoapi_ignore = ["*cli*", "*__main__.py"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "torch": ("https://pytorch.org/docs/stable", None),
    "PIL": ("https://pillow.readthedocs.io/en/stable/", None),
    "openslide": ("https://openslide.org/api/python/", None),
}

# For editing the pages.
html_context = {
    "github_user": "SBU-BMI",
    "github_repo": "wsinfer",
    "github_version": "main",
    "doc_path": "docs",
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# Define the json_url for our version switcher.
json_url = "https://wsinfer.readthedocs.io/en/latest/_static/switcher.json"

# Copied (with love) from conf.py of pydata-sphinx-theme.
# Define the version we use for matching in the version switcher.
version_match = os.environ.get("READTHEDOCS_VERSION")
# If READTHEDOCS_VERSION doesn't exist, we're not on RTD
# If it is an integer, we're in a PR build and the version isn't correct.
if not version_match or version_match.isdigit():
    # For local development, infer the version to match from the package.
    release = wsinfer.__version__
    if "dev" in release or "rc" in release:
        version_match = "dev"
        # We want to keep the relative reference if we are in dev mode
        # but we want the whole url if we are effectively in a released version
        json_url = "_static/switcher.json"
    else:
        version_match = "v" + release

html_theme = "pydata_sphinx_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Many of these options are copied directly from pydata-sphinx-theme's conf.py.
html_theme_options = {
    "logo": {
        "text": "WSInfer",
        "image_light": "_static/logo.svg",
        "image_dark": "_static/logo.svg",
        "alt_text": "WSInfer Logo",
    },
    "use_edit_page_button": True,
    "show_toc_level": 1,
    # "navbar_align": "content",
    "github_url": "https://github.com/SBU-BMI/wsinfer",
    "navbar_end": ["version-switcher", "theme-switcher", "navbar-icon-links"],
    "show_version_warning_banner": True,
    "switcher": {
        "json_url": json_url,
        "version_match": version_match,
    },
    "external_links": [
        {"name": "GitHub Repository", "url": "https://github.com/SBU-BMI/wsinfer"},
    ],
    "header_links_before_dropdown": 6,
}
# html_logo = "_static/logo.svg"
# html_favicon = "_static/log
