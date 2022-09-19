from skbuild import setup
from os.path import join, exists, normpath
from os import listdir, mkdir
from shutil import rmtree
from pathlib import Path

from openpiv_cxx._build_utilities.build_libs import (
    check_cmake_txt,
    modify_cmake_txt
)

def main():
    current_full_path = Path().absolute()
    external_dir = join(current_full_path, "extern")
    openpiv_cxx_dir = join(external_dir, "openpiv-c--qt")
    
    # modify cmake script based on our needs
    txt_to_mod = '  set(TIFF_EXTERNAL_LIBRARY_CXX ${OPENPIV_CXX_DIR}/external/libtiff/4.0.10)\n'
    
    dir_to_modify = normpath(join(openpiv_cxx_dir, "openpiv/CMakeLists.txt"))
    
    line_to_modify = 24
    
    if check_cmake_txt(dir_to_modify, line_to_modify, txt_to_mod) == False:
        print("Modifying openpiv-c--qt CMakeLists.txt to suite current build")
        modify_cmake_txt(dir_to_modify, line_to_modify, txt_to_mod)
    
    # remove existing builds
    if exists("_skbuild") == True:
        print("Found previous _skbuild build. Removing folder")
        rmtree("_skbuild")
    
    setup(
        name="OpenPIV-Python-cxx",
        description="DPIV software for pre/post processing and analysis of image pairs.",
        version="0.2.4",
        license="GPLv3",
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
        zip_safe=False
    )

if __name__ == "__main__":
    main()