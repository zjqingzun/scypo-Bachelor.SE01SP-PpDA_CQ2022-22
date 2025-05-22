#!/bin/bash


echo "[scripts/cenv.sh]     Executing ..."


# Check if Python is installed
echo "[scripts/cenv.sh]     Checking for Python installation ..."
python --version

echo "[scripts/cenv.sh]     Check if pip is installed ..."
pip --version

echo "[scripts/cenv.sh]     Downloading libraries ..."
pip install -r requirements.txt 

