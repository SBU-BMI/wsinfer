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
# Many of these options are copied directly from pydata-sphinx-theme's conf.py.
html_theme_options = {
    "logo": {
        "text": "WSInfer",
        # "image_dark": "_static/logo-dark.svg",
        # "alt_text": "WSInfer",
    },
    "use_edit_page_button": True,
    "show_toc_level": 1,
    "navbar_align": "left",  # [left, content, right] For testing that the navbar items align properly
    "navbar_center": ["version-switcher", "navbar-nav"],
    "show_version_warning_banner": True,
    # "show_nav_level": 2,
    # "navbar_start": ["navbar-logo"],
    # "navbar_end": ["theme-switcher", "navbar-icon-links"],
    # "navbar_persistent": ["search-button"],
    # "primary_sidebar_end": ["custom-template.html", "sidebar-ethical-ads.html"],
    # "article_footer_items": ["test.html", "test.html"],
    # "content_footer_items": ["test.html", "test.html"],
    # "footer_start": ["test.html", "test.html"],
    # "secondary_sidebar_items": ["page-toc.html"],  # Remove the source buttons
    "switcher": {
        "json_url": json_url,
        "version_match": version_match,
    },
}
# html_logo = "_static/logo.svg"
# html_favicon = "_static/log
