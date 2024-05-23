"""A setuptools based setup module.
See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

import setuptools as st
from codecs import open
from os import path
import numpy as np
from Cython.Build import cythonize


ext_modules = cythonize(
    [
        'sed_scores_eval/base_modules/cy_detection.pyx',
        'sed_scores_eval/base_modules/cy_medfilt.pyx',
     ]
)

here = path.abspath(path.dirname(__file__))

# Get the long description from the relevant file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

st.setup(
    name='sed_scores_eval',
    version='0.0.4',
    description='(Threshold-Independent) Evaluation of Sound Event Detection Scores',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/fgnt/sed_scores_eval',
    author='Department of Communications Engineering, Paderborn University',
    author_email='sek@nt.upb.de',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    keywords='sound recognition evaluation from classification scores',
    packages=st.find_packages(exclude=['contrib', 'docs', 'tests*']),
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'pathlib',
        'lazy_dataset',
        'einops',
        'sed_eval',
        'Cython',
    ],
    extras_require={
        'dev': ['check-manifest'],
        'test': ['coverage', 'jupyter', 'matplotlib'],
    },
    ext_modules=ext_modules,
    package_data={'sed_scores_eval': ['**/*.pyx']},  # https://stackoverflow.com/a/60751886
    include_dirs=[np.get_include()],
)
