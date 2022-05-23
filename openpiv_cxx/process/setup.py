def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    
    # create library
    #from CMakeSetup import create_lib_ext, return_outFolderName
    #outFolder = return_outFolderName()
    # create_lib_ext()
    
    config = Configuration(
        'process', 
        parent_package, 
        top_path
    )
    
    config.add_data_dir('_libs')
    
    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
