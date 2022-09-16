from skbuild import setup
from os.path import join, exists, normpath
from os import listdir, mkdir
from shutil import rmtree

from openpiv_cxx._build_utilities.vcpkg_get_triplet import return_cxx_triplet
from openpiv_cxx._build_utilities.build_libs import build_openpivcore

def main():
    current_full_path = ""
    external_dir = join(current_full_path, "extern")
    openpiv_cxx_dir = join(external_dir, "openpiv-c--qt")
    
    _libs_path = "_libs"
    if exists(_libs_path) != True:
        print("Warning: could not locate _libs folder\n" + 
              "Creating _libs folder and libraries")
        
        mkdir(_libs_path)        
        build_openpivcore(openpiv_cxx_dir, _libs_path)
    
    if exists("_skbuild") == True:
        print("Found previous _skbuild build. Removing folder")
        rmtree("_skbuild")
    
    python_min_version = 3.6
    
    req_dps = [
        'numpy',
        'imageio', 
        'pytest'
    ]
    req_py = ">={}".format(python_min_version)
    
    return_cxx_triplet(openpiv_cxx_dir)
    raise Excpetion("Terminatingn process")
    
    setup(
        name="OpenPIV-cxx",
        description="OpenPIV-Python with c++ backend",
        version="0.2.4",
        license="GPLv3",
        install_requires=req_dps, 
        python_requires=req_py,
        packages=[
            'openpiv_cxx',
            'openpiv_cxx.input_checker',
            'openpiv_cxx.interpolate',
            'openpiv_cxx.process',
            'openpiv_cxx.preprocess',
            'openpiv_cxx.validation',
            'openpiv_cxx.smooth',
            'openpiv_cxx.tools',
            'openpiv_cxx.windef'
        ],
        cmake_args=[
            f"-DTRIPLET_TO_USE={ return_cxx_triplet(openpiv_cxx_dir) }"
        ],
        zip_safe=False
    )

if __name__ == "__main__":
    main()