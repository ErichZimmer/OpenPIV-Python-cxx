<<<<<<< HEAD
import openpiv_cxx
from openpiv_cxx import *
=======

import openpiv_cxx
>>>>>>> 2eefe067e21993ca4875c9ed4dce6e913b6c6417

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
