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

from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "src._cython.cythomp",                 # <-- import name
        ["src/_cython/cythomp.pyx"],           # <-- relative file path
        include_dirs=[np.get_include()],
        # extra_compile_args=[...]  # if you need any
    )
]

setup(
    ext_modules=cythonize(
        extensions,
        language_level="3",
    ),
)