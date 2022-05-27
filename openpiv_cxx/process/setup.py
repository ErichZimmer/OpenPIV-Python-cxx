def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    from pathlib import Path
    from os.path import exists, join
    
    path_to_libs = join(Path(__file__).parent.absolute(), '_libs').replace("\\", "/") 
    
    if not exists(path_to_libs):
        raise Exception(
            "Could not locate '_libs' folder." +
            "\nPlease set current directory to '{}' and run 'python CMakeSetup.py build'".format(
                str(Path(__file__).parent.absolute()).replace("\\", "/")
            ) 
        )
        
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
