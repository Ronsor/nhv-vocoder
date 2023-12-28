#!/usr/bin/env python

import os

from setuptools import find_packages, setup

if os.path.exists("README.md"):
    README = open("README.md").read()
else:
    README = ""  # a placeholder, readme is generated on release
CHANGES = open("CHANGES.md").read()

setup(
    name="neural-homomorphic-vocoder",
    version="0.0.13",
    description="Pytorch implementation of neural homomorphic vocoder",
    url="https://github.com/k2kobayashi/neural-homomorphic-vocoder",
    author="K. KOBAYASHI",
    packages=find_packages(exclude=["examples", "test"]),
    long_description=(README + "\n" + CHANGES),
    long_description_content_type='text/markdown',
    license="MIT",
    install_requires=open("tools/requirements.txt").readlines(),
)
