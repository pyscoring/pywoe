# -*- coding: utf-8 -*-
"""
Package setup.
"""

import setuptools

# Load content saved elsewhere needed in the setup.
with open("README.md", "r") as fh:
    long_description = fh.read()

with open("pywoe/VERSION", "r") as fh:
    version = fh.read()

with open("requirements.txt", "r") as fh:
    requirements = fh.readlines()

setuptools.setup(
    name="pywoe",
    version=version,
    author="pyscoring",
    author_email="tadas.krisciunas@gmail.com",
    description="The missing scikit-learn addition to work with Weight-of-Evidence scoring.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pyscoring/pywoe",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3"
    ],
    python_requires='>=3.6',
)