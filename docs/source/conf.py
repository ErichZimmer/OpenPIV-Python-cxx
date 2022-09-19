import sys, os

import openpiv_cxx

sys.path.append(os.path.abspath("../../openpiv_cxx"))

# -- Project information -----------------------------------------------------
project = 'OpenPIV-Python-cxx'
copyright = '2022, OpenPIV Contributors'
author = 'OpenPIV Contributors'
release = '0.2.4'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.coverage',
    'sphinx.ext.napoleon',
#    'numpydoc',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '.ipynb_checkpoints']

# -- Options for HTML output -------------------------------------------------
html_theme = 'alabaster'
html_static_path = ['_static']