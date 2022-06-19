def return_cxx_triplet(path_to_pkg = "openpiv-c--qt/external/vcpkg/packages"):
    from os import listdir
    import pathlib
    
    openpiv_cxx_dir = str(str(pathlib.Path.home())).replace('\\', '/') + '/' + path_to_pkg
    test_file = listdir(openpiv_cxx_dir)[0]
    
    return test_file.split('_')[1]