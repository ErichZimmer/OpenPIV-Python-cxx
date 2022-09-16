Installation
============

Build from source
-----------------

To build the package, first you need to satisfy the requirements for vcpkg. Here is a general requirements list:
 + a compiler (e.g. MSVC 2019 or on UNIX, apt install build-essentials)
 + cmake
 + git (could be installed with conda or pip)
 + (UNIX) pkg-config (apt install pkg-config)
 + (UNIX) curl, zip, unzip, tar (apt install curl, zip, unzip, tar)
 + (UNIX) ninja (apt install ninja-build)

Next, dowload and install a python environment manager, such as miniconda. Setup a virtual environment and activate it  (with conda, use `conda create --name <env name> python=3.8`). You can now install git through conda or pip (unless you want to install it differently). When using git to clone this repository, you must clone it recursively due to third-party packages used in this repository. So when cloning, use `git clone --recursive https://github.com/ErichZimmer/OpenPIV-Python-cxx.git`.

When building this package, set your current directory to this package in the terminal used to compile this package (e.g. cd ...). Use 

    python setup.py build
    
to validate everything works (not necessary, just to prevent pollutting your virtual environment). If no errors are generated, then you can proceed with 

    python -m setup.py install
    
or 

    pip install .
    
If you run into any issues, please create an issue so it can be resolved.