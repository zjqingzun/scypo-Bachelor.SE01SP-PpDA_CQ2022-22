#!/bin/bash


echo "[scripts/start.sh]    Executing ..."


# Check if Python is installed
chmod +x scripts/cenv.sh
./scripts/cenv.sh

echo "[scripts/start.sh]    Running the application ..."
clear

# Initialize variables and set defaults
practice=""
while true; do
  case "$1" in
    -d|--default)
      python ./src/app-dp.py
      shift
      ;;
    -p|--practice)
      practice="$2"
      if [ "$practice" == "1" ]; then
        python ./src/dataset.py
        exit 0
      elif [ "$practice" == "2" ]; then
        python ./src/pre-processing.py
        exit 0
      else
        echo "Invalid practice number. Use 1 or 2 or 3."
        exit 0
      fi
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 [-d | --default] [-p PRACTICE | --practice PRACTICE] [-h|--help]"
      exit 0
      ;;
  esac
done

