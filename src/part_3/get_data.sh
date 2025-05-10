#!/bin/sh
PROJECT_DIR="/spark-examples/spark-project2/big_data_project_2"
DATA_DIR="$PROJECT_DIR/data"
UCI_DATA_URL1="https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
UCI_DATA_URL2="https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
OUTPUT_FILE1="train3.data"
OUTPUT_FILE2="test3.test"

echo "Creating data directory if not already present..."
mkdir -p "$DATA_DIR"
cd "$DATA_DIR" || { echo "Failed to access data directory"; exit 1; }

echo "Downloading UCI Adult Income dataset..."
curl -o "$OUTPUT_FILE1" "$UCI_DATA_URL1"
curl -o "$OUTPUT_FILE2" "$UCI_DATA_URL2"

echo "Download complete: saved as $DATA_DIR/$OUTPUT_FILE1"
echo "Download complete: saved as $DATA_DIR/$OUTPUT_FILE2"
