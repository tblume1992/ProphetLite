# -*- coding: utf-8 -*-
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ProphetLite",
    version="0.0.2",
    author="Tyler Blume",
    url="https://github.com/tblume1992/ProphetLite",
    long_description=long_description,
    long_description_content_type="text/markdown",
    description = "Like Prophet but using Numba and LASSO.",
    author_email = 'tblume@mail.USF.edu', 
    keywords = ['forecasting', 'time series', 'seasonality', 'trend'],
      install_requires=[           
                        'numpy',
                        'matplotlib',
                        'numba'
                        ],
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
