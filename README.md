| Platforms | Statistics | Health |
| --------- | ---------- | ------ |
| [![Windows](https://github.com/ErichZimmer/OpenPIV-Python-cxx/actions/workflows/build_windows.yml/badge.svg)](https://github.com/ErichZimmer/OpenPIV-Python-cxx/actions/workflows/build_windows.yml) | ![License](https://img.shields.io/github/license/ErichZimmer/OpenPIV-Python-cxx) | [![Documentation Status](https://readthedocs.org/projects/openpiv-python-cxx/badge/?version=latest)](https://openpiv-python-cxx.readthedocs.io/en/latest/?badge=latest) |
| [![macOS](https://github.com/ErichZimmer/OpenPIV-Python-cxx/actions/workflows/build_macos.yml/badge.svg)](https://github.com/ErichZimmer/OpenPIV-Python-cxx/actions/workflows/build_macos.yml) | ![Issues](https://img.shields.io/github/issues/ErichZimmer/OpenPIV-Python-cxx) | [![Build wheels](https://github.com/ErichZimmer/OpenPIV-Python-cxx/actions/workflows/build_wheels.yml/badge.svg)](https://github.com/ErichZimmer/OpenPIV-Python-cxx/actions/workflows/build_wheels.yml) |
| [![Linux](https://github.com/ErichZimmer/OpenPIV-Python-cxx/actions/workflows/build_linux.yml/badge.svg)](https://github.com/ErichZimmer/OpenPIV-Python-cxx/actions/workflows/build_linux.yml) | ![Stars](https://img.shields.io/github/stars/ErichZimmer/OpenPIV-Python-cxx) |

# OpenPIV

OpenPIV consists of Python modules for scripting and executing the analysis of 
a set of PIV image pairs. 

## Warning

The OpenPIV Python-c++ version is still in its *developmental* state. This means that there
are a few bugs and the API may change. 

## Building from source
### Setup environment
To build the package, first you need to satisfy the requirements for vcpkg. Here is a general requirements list:
 + a compiler (e.g. MSVC 2019 or on UNIX, apt install build-essentials)
 + cmake
 + git (could be installed with conda or pip)
 + (UNIX) pkg-config (apt install pkg-config)
 + (UNIX) curl, zip, unzip, tar (apt install curl, zip, unzip, tar)
 + (UNIX) ninja (apt install ninja-build)

Next, dowload and install a python environment manager, such as miniconda or edm.

Setup a virtual environment and activate it (with conda, use conda create --name <env name> python=3.8), (with edm is edm environments create <env name> --version=3.8 and then edm shell -e <env name>)

You can now install git through conda or pip (unless you want to install it differently). When using git to clone this repository, you must clone it recursively due to third-party packages used in this repository. So when cloning, use git clone --recursive https://github.com/ErichZimmer/OpenPIV-Python-cxx.git.

### To build:
When building this package, set your current directory to this package in the terminal used to compile this package (e.g. cd ...). We need to install the build requirements and upgrade pip. To do this, execute the folowing line:
```python 
pip install --upgrade -r requirements/build.txt
```

Next, we can build the actual package with

```python
python setup.py install .
``` 

or

```python
pip install .
``` 
or if the compilation fails due to build isolation,

```python
pip install --no-build-isolation .
``` 

### Optional dependencies
To further increase accuracy and performance, some functions utilize extra third-party packages. For instance, the smoothing algorithm implented by references 1 and 2 use SciPy minimization functions. Here are some optional, but not needed dependencies:
 - scipy : all-in-one post processing algorithm (smoothn)
 - matplotlib  : data visualization and creation of publication-ready plots
 - ffmpeg : loading and creating movies

These can be installed using `pip install openpiv_cxx[full]`.
 
### Optional dependencies
Documentation is inspired by [openpiv_tk_gui](https://github.com/OpenPIV/openpiv_tk_gui)
Find the [detailed documentation on readthedocs.io](https://openpiv-python-cxx.readthedocs.io/en/latest/index.html)

## To-do:
 - [ ] Full compatability with OpenPIV-Python (and subsequently openpiv_tk_gui)
 - [ ] Contour finding for mask images
 - [ ] dynamic/algorithmic masking
 - [ ] two-phase separation
 - [ ] FFTW double/single precision for much faster processing
 - [ ] Error correlation-based correction algorithm
 - [ ] Repeated correlation
 - [ ] 2D NxN subpixel centroid approximation
 - [ ] Image dewarping and transformations
 - [ ] spatial correlation (direct cross-correlation)
 - [ ] 5 point gaussian subpixel approximation
 - [ ] 2D 3x3 least squares gaussian subpixel approximation 
 - [ ] 3D PIV?
 - [ ] PIV guided PTV?
 - [ ] transcribe repository to numba and pythran

## Contributors

1. [Erich Zimmer](https://github.com/ErichZimmer)
2. [Tim Dewhirst](https://github.com/timdewhirst)
3. [Alex Liberzon](http://github.com/alexlib)

Copyright statement: `smoothn.py` is a Python version of `smoothn.m` originally created by D. Garcia [https://de.mathworks.com/matlabcentral/fileexchange/25634-smoothn], written by Prof. Lewis and available on Github [https://github.com/profLewis/geogg122/blob/master/Chapter5_Interpolation/python/smoothn.py]. We include a version of it in the `openpiv` folder for convenience and preservation. We are thankful to the original authors for releasing their work as an open source. OpenPIV license does not relate to this code. Please communicate with the authors regarding their license. 
## References
 - Garcia, D. (2010). Robust smoothing of gridded data in one and higher dimensions with missing values. Computational Statistics & Data Analysis, 54(4), 1167–1178. Elsevier BV. https://doi.org/10.1016%2Fj.csda.2009.09.020

 - Garcia, D. (2010). A fast all-in-one method for automated post-processing of PIV data. Experiments in Fluids, 50(5), 1247–1259. Springer Science and Business Media LLC. https://doi.org/10.1007%2Fs00348-010-0985-y

 - Kim, B.J., Sung, H.J. (2006). A further assessment of interpolation schemes for window deformation in PIV. Exp Fluids 41, 499–511. https://doi.org/10.1007/s00348-006-0177-y

 - Liberzon, A., Käufer, T., Bauer, A., Vennemann, P., & Zimmer, E. (2022). OpenPIV/openpiv-python: OpenPIV-Python v0.23.4. Zenodo. Retrieved 3 July 2022, from https://zenodo.org/record/4409178#.YsE9ouzMKM8.

 - Raffel M, Willert CE, Kompenhans J (1998) Particle image velocimetry: a practical guide, Springer, Berlin Heidelberg New York, pp 177-188

 - Scarano, Fulvio. (2001). Iterative image deformation methods in PIV. Measurement Science and Technology. 13. R1. 10.1088/0957-0233/13/1/201. 
 
 - Stein, D. (n.d.). Dbstein/FAST_INTERP: Numba accelerated interpolation on regular grids in 1, 2, and 3 dimensions. GitHub. Retrieved September 2, 2022, from https://github.com/dbstein/fast_interp 

 - Taylor, Z., Gurka, R., Kopp, G., & Liberzon, A. (2010). Long-Duration Time-Resolved PIV to Study Unsteady Aerodynamics. IEEE Transactions On Instrumentation And Measurement, 59(12), 3262-3269. https://doi.org/10.1109/tim.2010.2047149
 
 - Wikipedia contributors. (2022, April 10). Bilinear interpolation. In Wikipedia, The Free Encyclopedia. Retrieved 1 July 2022, from https://en.wikipedia.org/wiki/Bilinear_interpolation
 
 - Wikipedia contributors. (2022, March 18). Whittaker-Shannon interpolation formula. In Wikipedia, The Free Encyclopedia. Retrieved 30 June 2022, from https://en.wikipedia.org/w/index.php?title=Whittaker%E2%80%93Shannon_interpolation_formula&oldid=1077909297
