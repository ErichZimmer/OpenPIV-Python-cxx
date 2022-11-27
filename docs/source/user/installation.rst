============
Installation
============

Installing OpenPIV-Python-cxx
=============================

The **OpenPIV-cxx** library can be installed via pip, or compiled from source.

Install via pip (not yet available)
-----------------------------------

The code below will install **OpenPIV-cxx** from [PyPI](https://pypi.org/project/openpiv-cxx/)

.. code-block:: bash

    pip install openpiv-cxx

Compile from source
-------------------

The following are **necessary** for building and installing this package from source:

 - A C++17-compliant compiler
 - Python (>=3.6)
 - setuptools
 - scikit-build
 - Ninja
 - CMake
 - NumPy
 - ImageIO

Setup environment
+++++++++++++++++

To build the package, first you need to satisfy the requirements for vcpkg. Here is a general requirements list:

    a compiler (e.g. MSVC 2019 or on UNIX, apt install build-essentials)
    cmake
    git (could be installed with conda or pip)
    (UNIX) pkg-config (apt install pkg-config)
    (UNIX) curl, zip, unzip, tar (apt install curl, zip, unzip, tar)
    (UNIX) ninja (apt install ninja-build)

Next, dowload and install a python environment manager, such as miniconda or edm.

Setup a virtual environment and activate it (with conda, use conda create --name python=3.8), (with edm is edm environments create --version=3.8 and then edm shell -e )

You can now install git through conda or pip (unless you want to install it differently). When using git to clone this repository, you must clone it recursively due to third-party packages used in this repository. So when cloning, use git clone --recursive https://github.com/ErichZimmer/OpenPIV-Python-cxx.git.

Building and installing
+++++++++++++++++++++++

When building this package, set your current directory to this package in the terminal used to compile this package (e.g. cd ...). We need to install the build requirements and upgrade pip.

For pip:

.. code-block:: bash

    pip install --upgrade -r requirements/build.txt

or for conda users:

.. code-block:: bash

    conda install -c conda-forge pip setuptools cmake ninja scikit-build wheel

Next, we can build the actual package with

.. code-block:: bash

    python setup.py install

or

.. code-block:: bash

    pip install .

or if the compilation fails due to build isolation,

.. code-block:: bash

    pip install --no-build-isolation .


Unit Tests
==========

The unit tests for **OpenPIV-cxx** are included in the repository and uses the Python package :mod:`pytest`. 
To run the unit tests:

.. code-block:: bash

    # Run tests from the tests directory
    cd tests
    python -m pytest .

Documentation
=============

The documentation for **OpenPIV-cxx** is `hosted online at ReadTheDocs.

Building the documentation
--------------------------

The following are **required** for building documentation:

 - Sphinx
 - Read the Docs Sphinx Theme
 - nbsphinx 
 - jupyter_sphinx 

You can install these dependencies using conda:

.. code-block:: bash

    conda install -c conda-forge sphinx sphinx_rtd_theme nbsphinx jupyter_sphinx 

or pip:

.. code-block:: bash

    pip install sphinx sphinx-rtd-theme nbsphinx jupyter-sphinx

To build the documentation, run the following commands in the source directory:

.. code-block:: bash

    cd docs
    make html

To build a PDF of the documentation (requires LaTeX and/or PDFLaTeX):

.. code-block:: bash

    cd docs
    make latexpdf
