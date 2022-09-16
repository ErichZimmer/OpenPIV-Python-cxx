def return_cxx_triplet(openpivcore_dir):
    from os import listdir
    from os.path import join, normpath

    path_to_pkg = normpath(join(openpivcore_dir, "external/vcpkg/packages"))
    print("\n\n", path_to_pkg, "\n\n")
    test_file = listdir(path_to_pkg)[0]

    return test_file.split("_")[1]
