from Cython.Build import cythonize
from distutils.core import setup
from distutils.extension import Extension

import numpy as np
from setuptools import setup
install_requires = [
    'chainer>=1.16.0',
    'filelock',
    'nose',
    'numpy>=1.9.0',
    'protobuf',
    'six>=1.9.0',
]

try:
    numpy_include = np.get_include()
except AttributeError:
    numpy_include = np.get_numpy_include()

ext_modules = [
    Extension(
        "cython_bbox",
        ["deel/model/librcnn/bbox.pyx"],
        include_dirs=[numpy_include]
    ),
    Extension(
        "cpu_nms",
        ["deel/model/librcnn/cpu_nms.pyx"],
        include_dirs=[numpy_include]
    )
]


setup(
    name = "deel",
    version = "0.0.2",
    author = "UEI corporation",
    author_email="info@uei.co.jp",
    description = ("Deel; A High level deep learning description language"),
    license = "MIT",
    keywords = "chainer",
    url = "https://github.com/uei/deel",
    packages=[  'deel',
                'deel.network',
                'deel.model',
                'deel.model.librcnn'
                ],
    ext_modules=cythonize(ext_modules)
)
