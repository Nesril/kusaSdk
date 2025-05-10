from setuptools import setup, find_packages
import os

NAME = 'kusa'
DESCRIPTION = 'Kusa SDK: Securely access, preprocess, and train machine learning models on datasets with enhanced privacy features.'
URL = 'https://github.com/Nesril/kusaSdk'
EMAIL = 'nesredinhaji715@gmail.com' 
AUTHOR = 'Nesredin  / HAWD Techs' 
REQUIRES_PYTHON = '>=3.7.0' 
VERSION = '0.0.6' 

REQUIRED = [
    'requests>=2.20.0',
    'pandas>=1.1.0',    
    'cryptography>=3.0',
    'numpy>=1.18.0',
    'nltk>=3.5',
    'joblib>=0.14.0',
    'scikit-learn>=0.23.0', 
    'python-dotenv>=0.10.0'
]

EXTRAS = {
    'tensorflow': ['tensorflow>=2.2.0'], 
    'pytorch': ['torch>=1.8.0'],        
    'all_ml': ['tensorflow>=2.2.0', 'torch>=1.8.0'],
}

# --- The rest you shouldn't have to touch too much :) ---
here = os.path.abspath(os.path.dirname(__file__))

try:
    with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION # Fallback if README is not found

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    project_slug = NAME.lower().replace("-", "_").replace(" ", "_")
    with open(os.path.join(here, project_slug, '__version__.py')) as f: # Assumes version in kusa/__version__.py
        exec(f.read(), about)
else:
    about['__version__'] = VERSION


setup(
    name=NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*", "docs", "examples"]),
    # If your package is a single module, use this instead of 'packages':
    # py_modules=['kusa'],

    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True, # Important if you have non-code files specified in MANIFEST.in
    license='MIT', # IMPORTANT: CHOOSE YOUR LICENSE! And create a LICENSE file.
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: MIT License', # CHANGE AS NEEDED
        'Development Status :: 3 - Alpha', # 3 - Alpha, 4 - Beta, 5 - Production/Stable
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Security :: Cryptography',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: Implementation :: CPython',
        'Operating System :: OS Independent',
    ],
    keywords='kusa sdk dataset machine-learning ai secure privacy data preprocessing nlp tabular',
    project_urls={
        'Documentation': 'http://kuusa.netlify.app/docs',
        'Source': 'https://github.com/Nesril/kusaSdk',
        'Tracker': 'https://github.com/Nesril/kusaSdk/issues',
    },
)