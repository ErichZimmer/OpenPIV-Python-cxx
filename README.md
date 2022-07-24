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
 - Activate virtual python environment and run `python setup.py install` or `pip install .`.


## Documentation


## Demo notebooks 


## Contributors

1. [Tim Dewhirst](https://github.com/timdewhirst)
2. [Erich Zimmer](https://github.com/ErichZimmer)
3. [Alex Liberzon](http://github.com/alexlib)

Copyright statement: `smoothn.py` is a Python version of `smoothn.m` originally created by D. Garcia [https://de.mathworks.com/matlabcentral/fileexchange/25634-smoothn], written by Prof. Lewis and available on Github [https://github.com/profLewis/geogg122/blob/master/Chapter5_Interpolation/python/smoothn.py]. We include a version of it in the `openpiv` folder for convenience and preservation. We are thankful to the original authors for releasing their work as an open source. OpenPIV license does not relate to this code. Please communicate with the authors regarding their license. 

## References
- Garcia, D. (2010). Robust smoothing of gridded data in one and higher dimensions with missing values. Computational Statistics & Data Analysis, 54(4), 1167–1178. Elsevier BV. https://doi.org/10.1016%2Fj.csda.2009.09.020

- Garcia, D. (2010). A fast all-in-one method for automated post-processing of PIV data. Experiments in Fluids, 50(5), 1247–1259. Springer Science and Business Media LLC. https://doi.org/10.1007%2Fs00348-010-0985-y

 - Liberzon, A., Käufer, T., Bauer, A., Vennemann, P., & Zimmer, E. (2022). OpenPIV/openpiv-python: OpenPIV-Python v0.23.4. Zenodo. Retrieved 3 July 2022, from https://zenodo.org/record/4409178#.YsE9ouzMKM8.
 
 - Taylor, Z., Gurka, R., Kopp, G., & Liberzon, A. (2010). Long-Duration Time-Resolved PIV to Study Unsteady Aerodynamics. IEEE Transactions On Instrumentation And Measurement, 59(12), 3262-3269. https://doi.org/10.1109/tim.2010.2047149
 
 - Wikipedia contributors. (2022, April 10). Bilinear interpolation. In Wikipedia, The Free Encyclopedia. Retrieved July 1, 2022, from https://en.wikipedia.org/wiki/Bilinear_interpolation
 
 - Wikipedia contributors. (2022, March 18). Whittaker-Shannon interpolation formula. In Wikipedia, The Free Encyclopedia. Retrieved June 30, 2022, from https://en.wikipedia.org/w/index.php?title=Whittaker%E2%80%93Shannon_interpolation_formula&oldid=1077909297
