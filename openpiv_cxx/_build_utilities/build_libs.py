def build_openpivcore(
    openpiv_cxx_dir: str,
    _libs_dir: str
) -> None:
            
    import subprocess
    
    from platform import system
    from glob import glob
    from shutil import copy
    from os.path import exists, join, normpath
    
    if exists(join(openpiv_cxx_dir, "build")) != True:
        print("Could not locate cmake build folder. Building openpivcore")
        
        # cmake doesn't like the Windows \\ convention?
        if system().lower() == "windows":
            openpiv_cxx_dir = openpiv_cxx_dir.replace('\\', '/')
            
        cmake_args = ["cmake", "-S .", "-B build",
                      "-DCMAKE_BUILD_TYPE=Release"]
        
        build_args = ["cmake",
                      "--build", "build"]
        
        if system().lower() == "windows":
            build_args += ["--config", "Release"]
        
        # build openpivcore
        subprocess.check_call(cmake_args, cwd = openpiv_cxx_dir)
        subprocess.run(build_args, cwd = openpiv_cxx_dir)
    
    # now that the library has been built, locate dynamic libraries and static libraries
    print("Locating dynamic and static libraries")
    if system().lower() == "windows":
        files = list(glob(
            normpath(
                join(openpiv_cxx_dir, "build/out/Release/*.dll"))
        ))
        files += list(glob(
            normpath(
                join(openpiv_cxx_dir, "build/openpiv/Release/*.lib"))
        ))
        
    elif system().lower() == "darwin":
        files = list(glob(
            normpath(
                join(openpiv_cxx_dir, "build/out/*.dylib"))
        ))
        
    elif system().lower() == "linux":
        files = list(glob(
            normpath(
                join(openpiv_cxx_dir, "build/out/*.so"))
        ))
    
    else:
        raise Exception(
            "Operating system type not found. Please raise issue on " +
            "main repository"
        )
    
    # Now that we (hopefully) located the files, copy them
    print("Copying dynamic and static libraries to _libs folder")
    
    for file in files:
        copy(file, _libs_dir)