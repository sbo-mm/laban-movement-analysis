from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

ext_modules = [
    Extension(
        "asfamcparser",
        ["AMCFileReader.pyx"],
        libraries=["m"],
        extra_compile_args = ["-ffast-math"]
    )
]

setup(
      name = "asfamcparser",
      include_dirs = [np.get_include()],
      cmdclass = {"build_ext": build_ext},
      ext_modules = ext_modules
)
