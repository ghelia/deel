from setuptools import setup
install_requires = [
    'chainer>=1.16.0',
    'filelock',
    'nose',
    'pillow',
    'h5py',
    'numpy>=1.9.0',
    'protobuf',
    'six>=1.9.0',
]
setup(
    name = "deel",
    version = "0.0.4.6",
    author = "UEI corporation",
    author_email="info@uei.co.jp",
    description = ("Deel; A High level deep learning description language"),
    license = "MIT",
    keywords = "chainer",
    url = "https://github.com/uei/deel",
    install_requires=install_requires,
    packages=[  'deel',
                'deel.network',
                'deel.model',
                ],
)
