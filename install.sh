#!/bin/bash

PYTHON=python



# create venv
echo "Creating venv..."
$PYTHON -m venv venv

# set PYTHON to python in venv
PYTHON=venv/bin/python

# upgrade pip
echo "Upgrading pip..."
$PYTHON -m pip install --upgrade pip

# install requirements
echo "Installing requirements..."
$PYTHON -m pip install -r requirements.txt

# Done?
echo "Done installing!"