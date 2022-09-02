from skbuild import setup
from setuptools import find_packages
from platform import system
from glob import glob
from os.path import join, exists
from os import listdir, mkdir

from openpiv_cxx._build_utilities.vcpkg_get_triplet import return_cxx_triplet
from openpiv_cxx._build_utilities.build_libs import build_openpivcore

import pathlib

def main():
    current_full_path = pathlib.Path().resolve()
    external_dir = join(current_full_path, "extern")
    openpiv_cxx_dir = join(external_dir, "openpiv-c--qt")
    
    if exists("_libs") != True:
        while True:
            answer = input("_libs folder not found. " + 
                           "Would you like to build openpivcore? ")
            
            if answer.lower() in ['n', "no"]:
                build = False; break
                
            elif answer.lower() in ['y', "yes"]:
                build = True; break
                
            else:
                print("Invalid response. Valid responses are 'n', 'no', 'y', and 'yes'")
        
        if build:
            _libs_path = join(current_full_path, "_libs")
            print(_libs_path)
            mkdir(_libs_path)
            
            build_openpivcore(openpiv_cxx_dir, _libs_path)
            
    # for cmake
    if system().lower() == "windows":
        openpiv_cxx_dir = openpiv_cxx_dir.replace('\\', '/')
    
    python_min_version = 3.6
    
    req_dps = [
        'numpy',
        'imageio', 
        'pytest'
    ]
    req_py = ">={}".format(python_min_version)

    setup(
        name="OpenPIV-cxx",
        description="OpenPIV-Python with c++ backend",
        version="0.2.0",
        license="GPLv3",
        install_requires=req_dps, 
        python_requires=req_py,
        packages=[
            'openpiv_cxx',
            'openpiv_cxx.interpolate',
            'openpiv_cxx.process',
            'openpiv_cxx.preprocess',
            'openpiv_cxx.validation',
            'openpiv_cxx.smooth',
            'openpiv_cxx.windef'
        ],
        cmake_args=[
            f"-DTRIPLET_TO_USE={ return_cxx_triplet(openpiv_cxx_dir) }",
            f"-DOPENPIV_CXX_DIR={ openpiv_cxx_dir }"
        ],
        zip_safe=False
    )

if __name__ == "__main__":
    main()