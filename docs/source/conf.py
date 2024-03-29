import openpiv_cxx
from openpiv_cxx import *

# -- Project information -----------------------------------------------------
project = "OpenPIV-Python-cxx"
copyright = "2022, OpenPIV Contributors"
author = "OpenPIV Contributors"
release = "0.3.5"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    #    'numpydoc',
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", ".ipynb_checkpoints"]

# -- Options for HTML output -------------------------------------------------
html_theme = "alabaster"
html_static_path = ["_static"]
