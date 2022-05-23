import tempfile

def has_flag(compiler, flagname):
    with tempfile.NamedTemporaryFile('w', suffix='.cpp') as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except CompileError:
            return False
    return True


def cpp_flag(compiler):
    """Return the -std=c++[11/14] compiler flag"""    
    if has_flag(compiler, '-std=c++14'):
        return '-std=c++14'
    elif has_flag(compiler, '-std=c++11'):
        return '-std=c++11'
    else:
        raise RuntimeError('Unsupported compiler -- at least C++11 support is needed!')
        
        
def pre_build_hook(build_ext, ext):
    cc = build_ext._cxx_compiler
    
    args = ext.extra_compile_args

    if cc.compiler_type == 'msvc':
        args.append('/EHsc')
    else:
        # Don't export library symbols
        if has_flag(cc, '-fvisibility=hidden'):
            args.append('-fvisibility=hidden')


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    from os.path import dirname, join
    from glob import glob
    import pybind11
    
    include_dirs = [pybind11.get_include(True), pybind11.get_include(False)]

    config = Configuration(
        'spatial_filters', 
        parent_package, 
        top_path
    )
    
    base_path = dirname(__file__)
    
    src_files = sorted(glob(join(base_path, "*.cpp")) + glob(join(base_path, "*.h"))) 
    
    ext = config.add_extension(
        'spatial_filters_cpp',
        sources=src_files,
        include_dirs=include_dirs,
        language='c++'
    )
    
    ext._pre_build_hook = pre_build_hook
    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
