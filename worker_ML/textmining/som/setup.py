from setuptools import setup
from setuptools import Extension
try:
    pass
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension

from Cython.Distutils import build_ext
import numpy


ext_modules = [Extension('csom',['csom.pyx'],
                        include_dirs=[numpy.get_include()],
                        )]

setup(name = "Hello World app",
        cmdclass = {'build_ext': build_ext},
        ext_modules = ext_modules)
