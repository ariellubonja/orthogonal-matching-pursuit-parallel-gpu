from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    # Changed file names bcs test I think its reserved. Seems like weird behavior
    ext_modules=cythonize("test_stuff.pyx"),
    include_dirs=[numpy.get_include()]
)