# coding=utf-8
"""
QuantNodes - AI-native quantitative research framework
"""

try:
    from importlib.metadata import PackageNotFoundError, version as _pkg_version
    try:
        __version__ = _pkg_version("quantnodes")
    except PackageNotFoundError:
        __version__ = "0.0.0+local"
except ImportError:
    __version__ = "0.0.0+local"

__author__ = 'sn0wfree'
