"""Build script for geoqc Cython extensions.

Usage:
    cd geocore
    PYTHONPATH=. python setup.py build_ext --inplace

This compiles geoqc/_exterior_cy.pyx -> geoqc/_exterior_cy.so
"""

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

extensions = [
    Extension(
        "geoqc._exterior_cy",
        sources=["geoqc/_exterior_cy.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3", "-march=native"],
    ),
]

setup(
    name="geoqc_cython",
    ext_modules=cythonize(extensions, language_level=3),
    zip_safe=False,
)
