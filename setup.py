# coding=utf-8
from setuptools import setup, find_packages

from Nodes import __version__, __author__

setup(
    name="Nodes",
    version=__version__,
    keywords=("Node", "Databases"),
    description="node programming",
    long_description=open('README.md', 'r'),
    license="MIT Licence",

    url="http://github.com/sn0wfree",
    author=__author__,
    author_email="snowfreedom0815@gmail.com",

    packages=find_packages(),
    include_package_data=True,
    platforms="any",
    install_requires=['numpy==1.16.4', 'pandas==0.25.1', 'PyMySQL==0.9.2', 'SQLAlchemy==1.2.16'],

    scripts=[],
    # entry_points={
    #     'console_scripts': [
    #         'test = test.help:main'
    #     ]
    # }
)
