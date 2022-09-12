def return_cxx_triplet(openpivcore_dir):
    from os import listdir
    from os.path import join

    path_to_pkg = join(openpivcore_dir, "external/vcpkg/packages")
    test_file = listdir(path_to_pkg)[0]

    return test_file.split("_")[1]
