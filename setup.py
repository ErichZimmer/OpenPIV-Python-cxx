from skbuild import setup
from setuptools import find_packages
from sys import executable
from glob import glob
from os.path import join

from openpiv_cxx._build_utilities.vcpkg_get_triplet import return_cxx_triplet

import platform
import pathlib
import pybind11

openpiv_cxx_dir = str(pathlib.Path.home()).replace("\\", "/") + '/openpiv-c--qt'
pybind11_dir = pybind11.get_cmake_dir().replace("\\", "/") 

if platform.system() == "Windows":
    dynamic_libs = glob(join(openpiv_cxx_dir, "build/out/Release/*.dll"))
    if len(dynamic_libs) < 3:
        raise Exception("Please build openpiv-c--qt with Release config")
else:
    pass

numpy_min_version = "1.22"
pybind11_min_version = "2.8" 
python_min_version = "3.8"

#req_dps = [
#    "numpy>={}".format(numpy_min_version),
#    "pybind11>={}".format(pybind11_min_version)
#    
#]

req_dps = ['numpy']
req_py = ">={}".format(python_min_version)
    
setup(
    name="OpenPIV-cxx",
    description="OpenPIV-Python with c++ backend",
    version="0.0.7",
    license="GPLv3",
    install_requires=req_dps, 
    python_requires=req_py,
    packages=[
        'openpiv_cxx',
        'openpiv_cxx.process',
        'openpiv_cxx.spatial_filters',
        'openpiv_cxx.smooth'
    ],
    cmake_args=[
        f"-DTRIPLET_TO_USE={ return_cxx_triplet() }",
        f"-DOPENPIV_CXX_DIR={ openpiv_cxx_dir }",
        f"-DPYBIND11_DIR={ pybind11_dir }",
        f"-DPYTHON_EXECUTABLE={ executable }"
    ],
    zip_safe=False
)
