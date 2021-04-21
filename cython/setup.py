from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy

# https://stackoverflow.com/questions/28301931/how-to-profile-cython-functions-line-by-line
# from Cython.Compiler.Options import get_directive_defaults
# directive_defaults = get_directive_defaults()
#
# directive_defaults['linetrace'] = True
# directive_defaults['binding'] = True
#
# extensions = [
#     Extension("test", ["test.pyx"], define_macros=[('CYTHON_TRACE', '1')])
# ]

setup(
    ext_modules=cythonize("test.pyx"),
    include_dirs=[numpy.get_include()],
)