import sys
from setuptools import setup


def getRequirements():
    with open("requirements.txt", "r") as f:
        read = f.read()

    return read.split("\n")


setup(
    name = 'K Means and K Means Plus Plus Classifier',
    version= "1.0.1",
    description='K Means and K Means Plus Plus for un marked data Classifications',
    long_description='using the reletavie distance between words we can infere if thery are from the same class or not',
    author='Mortar Defender',
    license='MIT License',
    url = '__',
    setup_requires = getRequirements(),
    install_requires = getRequirements(),
    include_package_data=True
)
