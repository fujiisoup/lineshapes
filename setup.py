#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import re
from setuptools import setup


# load version form _version.py
VERSIONFILE = "lineshapes/_version.py"
with open(VERSIONFILE, 'rt', encoding='utf-8') as f:
    verstrline = f.read()

VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError(f"Unable to find version string in {VERSIONFILE}.")

# README
with open('README.md', 'r', encoding='utf-8') as f:
    long_desc = f.read()

# module

setup(name='lineshapes',
      version=verstr,
      author="Keisuke Fujii",
      author_email="fujiisoup@gmail.com",
      description=("Python package to calculate line shapes."),
      long_description=long_desc,
      license="Apache License, Version 2.0",
      keywords="atomic physics, spectroscopy",
      url="https://github.com/fujiisoup/lineshapes",
      packages=["lineshapes"],
      package_dir={'lineshapes': 'lineshapes'},
      py_modules=['lineshapes.__init__'],
      test_suite='tests',
      requires=[
        'numpy',
        'py3nj'],
      classifiers=['License :: OSI Approved :: BSD License',
                   'Natural Language :: English',
                   'Operating System :: MacOS :: MacOS X',
                   'Operating System :: Microsoft :: Windows',
                   'Operating System :: POSIX :: Linux',
                   'Programming Language :: Python :: 3.6',
                   'Topic :: Scientific/Engineering :: Physics'],
      )

