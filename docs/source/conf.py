# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath("../../"))

# ruff: noqa

# -- Project information ----------------------------------------------------- #

project = "OpenVINO™ XAI"
copyright = "2024, Intel(R) Corporation"
author = "Intel(R) Corporation"
release = "1.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx_copybutton",
    "sphinx.ext.autosummary",  # Create neat summary tables
    "sphinx.ext.viewcode",  # Find the source files
    "sphinx.ext.autosectionlabel",  # Refer sections its title
    "sphinx.ext.intersphinx",  # Generate links to the documentation
    "sphinx_tabs.tabs",
    "sphinx_design",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["build", "Thumbs.db", ".DS_Store"]



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

#html_theme = 'sphinx_rtd_theme'
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_theme_options = {
#    #"navbar_center": [],
#    #"navbar_end": ["search-field.html", "theme-switcher.html", "navbar-icon-links.html"],
#    #"search_bar_text": "Search",
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/openvinotoolkit/openvino_xai",
            "icon": "_static/github_icon.png",
            "type": "local",
        },
    ],
}
html_sidebars = {
   '**': [
       'globaltoc.html',
   ],
}
html_css_files = [
    "custom.css",
]


# -- Extension configuration -------------------------------------------------
autodoc_mock_imports = ["bs4", "requests"]
autodoc_docstring_signature = True
autodoc_member_order = "bysource"
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}
autodoc_member_order = "groupwise"
autodoc_default_options = {
    "members": True,
    "methods": True,
    "special-members": "__call__",
    "exclude-members": "_abc_impl",
    "show-inheritance": True,
}

autoclass_content = "both"

autosummary_generate = True  # Turn on sphinx.ext.autosummary
autosummary_ignore_module_all = False  # Summary list in __all__ no others
# autosummary_imported_members = True # document classes and functions imported in modules
