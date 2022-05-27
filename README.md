# OpenPIV

OpenPIV consists of Python and c++ modules for scripting and executing the analysis of 
a set of PIV image pairs. 

## Warning

The OpenPIV Python-c++ version is still in its *developmental* state. This means that there
is a lot of bugs and the API may change. 

## Installing

    
### To build from source
 - Follow the instructions on [openpiv-c--qt](https://github.com/OpenPIV/openpiv-c--qt) and compile openpivcore.
 - Git clone this repository.
 - Activate virtual python environment and run `python setup.py install`.
 - If an error occurs about `_libs` folder not being found, that means that the CMake build was not executed. Run `python CMakeSetup.py build` at the directory specified by the error.


## Documentation


## Demo notebooks 


## Contributors

1. [Tim Dewhirst](https://github.com/timdewhirst)
2. [Erich Zimmer](https://github.com/ErichZimmer)
3. [Alex Liberzon](http://github.com/alexlib)
