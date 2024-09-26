from setuptools import setup, find_packages

setup(
    name='kusa',
    version='0.1.1',
    description='SDK for accessing purchased datasets',
    author='HAWD Techs',
    author_email='haws@gmail.com',
    packages=find_packages(),
    install_requires=[
        'requests',
        'pandas',
    ],
    python_requires='>=3.6',
)
