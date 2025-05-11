#!/bin/bash


echo "scripts/cenv.sh: Executing ..."


# Check if Python is installed
echo "Checking for Python installation ..."
python --version

echo "Check if pip is installed ..."
pip --version

echo "Downloading libraries ..."
pip install -r requirements.txt 

