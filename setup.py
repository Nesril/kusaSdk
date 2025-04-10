from setuptools import setup, find_packages

setup(
    name='kusa',
    version='0.1.8',
    description='SDK for accessing purchased datasets',
    author='HAWD Techs',
    author_email='hawd@gmail.com',
    packages=find_packages(),
    install_requires=[
        'requests',
        'pandas',
        'cryptography',
        'numpy',
        "nltk",
        "joblib",
        "scikit-learn"
    ],
    python_requires='>=3.6',
)
