# coding=utf-8
from setuptools import setup, find_packages

from QuantNodes import __version__, __author__

setup(
    name="nodes",
    version=__version__,
    keywords=("Node", "Databases", "Quantitative"),
    description="AI-native quantitative research framework with symbolic computation and SQL pushdown",
    long_description="QuantNodes is a quantitative research node architecture platform that implements unified BaseNode + Pipeline pattern for factor calculation, backtest analysis, and database queries.",
    license="MIT Licence",

    url="http://www.github.com/sn0wfree",
    author=__author__,
    author_email="snowfreedom0815@gmail.com",

    packages=find_packages(),
    include_package_data=True,

    entry_points={
        'console_scripts': [
            'quantnodes=QuantNodes.cli:main',
        ],
    },
    python_requires='>=3.9',
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
