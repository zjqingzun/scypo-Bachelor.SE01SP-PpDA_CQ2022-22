#!/bin/bash


echo "scripts/start.sh: Executing ..."


# Check if Python is installed
chmod +x scripts/cenv.sh
./scripts/cenv.sh

echo "Running the application ..."
python src/app-dp.py