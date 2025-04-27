from setuptools import setup, find_packages

setup(
    name='kusa',
    version='0.2.0',
    description='SDK for accessing purchased datasets',
    author='HAWD Techs',
    author_email='hawd@gmail.com',
    python="3.12.5",
    packages=find_packages(),
    install_requires=[
        'requests',
        'pandas',
        'cryptography',
        'numpy',
        "nltk",
        "joblib",
        "scikit-learn",
        "torch",
        "tensorflow"
    ],
    python_requires='>=3.6',
)
