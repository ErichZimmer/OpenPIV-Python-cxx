triplet = "x64-windows" # for locating vcpkg packages
outFolderName = "_libs" # for locating libraries

import os
import re
import subprocess
import sys
import platform
import pathlib
  
from setuptools import Extension
from setuptools.command.build_ext import build_ext

import pybind11

def return_outFolderName():
    return outFolderName

def return_cxx_triplet():
    path_to_packages = "openpiv-c--qt/external/vcpkg/packages"
    openpiv_cxx_dir = str(pathlib.Path.home()).replace('\\', '/') + '/' + path_to_packages
    test_file = os.listdir(openpiv_cxx_dir)[0]
    return test_file.split('_')[1]


# A CMakeExtension needs a sourcedir instead of a file list.
class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

        
class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: "
                + ", ".join(e.name for e in self.extensions)
            )

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        
        # required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep
        
        cfg = "Release"
        
        openpiv_cxx_dir = str(pathlib.Path.home()).replace("\\", "/") + '/openpiv-c--qt'
        pybind11_dir = pybind11.get_cmake_dir().replace("\\", "/") 
        
        _build_folder = os.path.join(pathlib.Path(__file__).parent.absolute(), outFolderName).replace("\\", "/") 
        
        try:
            triplet = return_cxx_triplet()
        except: 
            raise Exception(
                "Could not find triplet for vcpkg."
            )
            
        cmake_args = [
            f"-DLIB_COPY_DIR={_build_folder}",
            f"-DTRIPLET_TO_USE={triplet}",
            f"-DOPENPIV_CXX_DIR={openpiv_cxx_dir}",
            f"-DPYBIND11_DIR={pybind11_dir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            #f"-DCMAKE_BUILD_TYPE={cfg}",  # not used on MSVC, but no harm
        ]

        build_args = []
        build_args += ["--config", cfg]

        if platform.system() == "Windows":
            #cmake_args += [
            #    "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}".format(cfg.upper(), extdir)
            #]
            cmake_args += ["-A", "x64" if sys.maxsize > 2**32 else "Win32"]
            build_args += ["--", "/m"]
        else:
            cmake_args += ["-DCMAKE_BUILD_TYPE=" + cfg]
            build_args += ["--", "-j2"]

        env = os.environ.copy()
        env["CXXFLAGS"] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get("CXXFLAGS", ""), self.distribution.get_version()
        )
        
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
            
        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env
        )
        subprocess.check_call(
            ["cmake", "--build", "."]
            + build_args,
            cwd=self.build_temp,
        )


def create_lib_ext():
    from setuptools import setup
    
    # create libs folder so we can copy and paste dynamic libraries into it
    _build_folder = os.path.join(os.path.dirname(__file__), outFolderName)
    if not os.path.exists(_build_folder):
        os.makedirs(_build_folder)
        
    setup(
        name="_process",
        version="0.0.2",
        author="Erich Zimmer",
        # author_email="",
        description="Create wrapper extensions for OpenPIV-c++.",
        # long_description="",
        ext_modules=[CMakeExtension("_process")],
        cmdclass={"build_ext": CMakeBuild},
        zip_safe=False,
        python_requires=">=3.8",
    )
    
    
if __name__ == '__main__':
    create_lib_ext()