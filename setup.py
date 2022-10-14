from skbuild import setup
from os.path import join, exists, normpath
from os import listdir, mkdir
from shutil import rmtree
from pathlib import Path


def get_pkg_from_txt(dir_):
    with open(dir_, "r", encoding="utf-8") as file:
        data = file.readlines()
        data = [pkg.strip("\n") for pkg in data] 
    return data


def check_cmake_txt(dir_, line_num, line_mod):
    with open(dir_, "r", encoding="utf-8") as file:
        data = file.readlines()

    return data[line_num] == line_mod


def modify_cmake_txt(dir_, line_num, line_mod):
    with open(dir_, "r", encoding="utf-8") as file:
        data = file.readlines()

    data[line_num] = line_mod

    with open(dir_, "w", encoding="utf-8") as file:
        file.writelines(data)


def main():
    current_full_path = Path().resolve()
    external_dir = join(current_full_path, "extern")
    openpiv_cxx_dir = join(external_dir, "openpiv-c--qt")

    # modify cmake script based on our needs
    txt_to_mod = (
        "  set(TIFF_EXTERNAL_LIBRARY_CXX ${OPENPIV_CXX_DIR}/external/libtiff/4.0.10)\n"
    )

    dir_to_modify = normpath(join(openpiv_cxx_dir, "openpiv/CMakeLists.txt"))

    line_to_modify = 24

    if check_cmake_txt(dir_to_modify, line_to_modify, txt_to_mod) == False:
        print("Modifying openpiv-c--qt CMakeLists.txt to suite current build")
        modify_cmake_txt(dir_to_modify, line_to_modify, txt_to_mod)

    # remove existing builds
    if exists("_skbuild") == True:
        print("Found previous _skbuild build. Removing folder")
        rmtree("_skbuild")

    # get install requirements
    install_requires = get_pkg_from_txt("requirements/depend.txt")
    
    # get extras
    extras = {}
    extras["build"] = get_pkg_from_txt("requirements/build.txt")
    extras["full"] = get_pkg_from_txt("requirements/full.txt")
    extras["docs"] = get_pkg_from_txt("requirements/docs.txt") + extras["full"]
    extras["test"] = get_pkg_from_txt("requirements/test.txt") + extras["full"]
    
    setup(
        version="0.3.6",
        package_dir={
            "openpiv_cxx": "lib",
            "openpiv_cxx.filters": "lib/filters",
            "openpiv_cxx.input_checker": "lib/input_checker",
            "openpiv_cxx.inpaint_nans": "lib/inpaint_nans",
            "openpiv_cxx.interpolate": "lib/interpolate",
            "openpiv_cxx.openpiv": "lib/openpiv",
            "openpiv_cxx.process": "lib/process",
            "openpiv_cxx.validate": "lib/validate",
            "openpiv_cxx.smooth": "lib/smooth",
            "openpiv_cxx.tools": "lib/tools",
            "openpiv_cxx.windef": "lib/windef"
        },
        packages=[
            "openpiv_cxx",
            "openpiv_cxx.filters",
            "openpiv_cxx.input_checker",
            "openpiv_cxx.inpaint_nans",
            "openpiv_cxx.interpolate",
            "openpiv_cxx.openpiv",
            "openpiv_cxx.process",
            "openpiv_cxx.validate",
            "openpiv_cxx.smooth",
            "openpiv_cxx.tools",
            "openpiv_cxx.windef"
        ],
        cmake_languages=[
            "C",
            "CXX"
        ],
        install_requires=install_requires,
        extras_require=extras,
        zip_safe=False
    )


if __name__ == "__main__":
    main()
