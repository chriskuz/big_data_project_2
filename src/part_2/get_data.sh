#!/bin/sh
PROJECT_DIR="/spark-examples/spark-project2/big_data_project_2"
DATA_DIR="$PROJECT_DIR/data"
ZIP_NAME="framingham.zip"
KAGGLE_USER=$(jq -r .username ~/.kaggle/kaggle.json)
KAGGLE_KEY=$(jq -r .key ~/.kaggle/kaggle.json)


echo "Creating data dictionary if not already present.."
mkdir -p "$DATA_DIR"
cd "$DATA_DIR" || { echo: "Failed data directory access"; exit 1; }


echo "Downloading NBA shot logs ZIP from Kaggle"
curl -L -u "${KAGGLE_USER}:${KAGGLE_KEY}" \
  -o "$ZIP_NAME" \
  "https://www.kaggle.com/api/v1/datasets/download/aasheesh200/framingham-heart-study-dataset"


echo "Extracting contents"
unzip -o "$ZIP_NAME" 
rm -f "$ZIP_NAME" 

echo "Done"