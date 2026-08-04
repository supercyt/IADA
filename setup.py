"""Compatibility shim for tools that still expect a setup.py file.

Project metadata and dependencies live in pyproject.toml.
"""

from setuptools import setup


setup()
