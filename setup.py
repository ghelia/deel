from setuptools import setup
install_requires = [
    'chainer>=1.16.0',
    'filelock',
    'nose',
    'numpy>=1.9.0',
    'protobuf',
    'six>=1.9.0',
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
                ],
)
