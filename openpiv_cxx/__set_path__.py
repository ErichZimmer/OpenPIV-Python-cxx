import os
import sys

extra_dll_dir = os.path.join(os.path.dirname(__file__), '_libs')

if os.path.isdir(extra_dll_dir):
    os.environ.setdefault('PATH', '')
    os.environ['PATH'] += os.pathsep + extra_dll_dir
del os, sys