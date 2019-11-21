from setuptools import setup, find_packages
import sys

req_version = (3, 7)

long_description = """
Flaskanic is a simple web interface to a machine learning model of Titanic survivors 
based on the Flask micro-framework. Use your Titanic machine learning model 
to make predictions and display them as a html data table. Serve the model as 
an API and get predictions by sending a POST request with JSON data.
"""

if sys.version_info < req_version:
    sys.exit("Python 3.7 or higher required to run this code, " +
             sys.version.split()[0] + " detected, exiting.")

setup(
    name="flaskanic",
    version="0.1",
    author="Simon Carozza",
    author_email="simoncarozza@gmail.com",
    description="flask interface to ML model of Titanic survivors",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SimonCarozza/flaskanic",
    packages=find_packages(),
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Microsoft :: Windows :: Windows 10",
        "Operating System :: Other OS"
    ],
    install_requires=[
        "autolrn",
        "scikit-learn>=0.20.1",
        "pandas",
        "tensorflow",
        "flask"
    ],
    package_data={
        "flaskanic": [
            "datasets/*.csv", "datasets/*.zip"]})
