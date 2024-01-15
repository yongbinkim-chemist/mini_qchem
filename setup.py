import io

from setuptools import setup, find_packages

# This reads the __version__ variable from openfermionqchem/_version.py
exec(open('src/_version.py').read())

# Readme file as long_description:
# long_description = io.open('README.rst', encoding='utf-8').read()

# Read in requirements.txt
requirements = open('requirements.txt').readlines()
requirements = [r.strip() for r in requirements]

setup(
    name='mini_qchem',
    version=__version__,
    author='Yongbin Kim',
    author_email='chem.yongbin@gmail.com',
    # url='http://www.iopenshell.usc.edu',
    description='Cute Electronic Structure Program',
    # long_description=long_description,
    install_requires=requirements,
    license='Apache 2',
    packages=find_packages()
)
