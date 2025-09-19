from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "src.cython.test",                 # <-- import name
        ["src/cython/test.pyx"],           # <-- relative file path
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