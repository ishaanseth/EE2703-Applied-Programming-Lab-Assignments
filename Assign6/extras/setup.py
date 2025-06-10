# setup.py

from setuptools import setup
from Cython.Build import cythonize
import numpy

# Configure the setup for building the Cython extension
setup(
    # cythonize compiles the cy_trapz.pyx file into a C extension
    ext_modules=cythonize(
        "cy_trapz.pyx",
        compiler_directives={'language_level': "3"}  # Set Python language level to 3
    ),
    # Include NumPy headers for efficient array handling in Cython
    include_dirs=[numpy.get_include()],
)
