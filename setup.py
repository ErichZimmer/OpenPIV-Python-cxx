def configuration(parent_package='',top_path=None):
    import os
    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')
        
    from numpy.distutils.misc_util import Configuration
    config = Configuration(None,parent_package,top_path)
    config.set_options(
        ignore_setup_xxx_py=True,
        assume_default_configuration=True,
        delegate_options_to_subpackages=True,
        quiet=True
    )
    
    config.add_subpackage('openpiv_cxx')
        
    return config


def setup_package():
    from distutils.command.sdist import sdist
    
    numpy_min_version = "1.22"
    pybind11_min_version = "2.8" 
    python_min_version = "3.8"
    
    req_dps = [
        "numpy>={}".format(numpy_min_version),
        "pybind11>={}".format(pybind11_min_version)
        
    ]
    
    req_py = ">={}".format(python_min_version)
    
    metadata = dict(
        name="openpiv_cxx",
        maintainer="OpenPIV Developers",
        #maintainer_email="",
        #url="",
        #download_url="",
        #project_url="",
        liscense="GPLv3",
        install_requires = req_dps,
        python_requires = req_py,
        cmdclass={'sdist': sdist},
        zip_safe=False
    )
    

if __name__ == '__main__':
    from numpy.distutils.core import setup
    
    setup(**configuration(top_path='').todict())
