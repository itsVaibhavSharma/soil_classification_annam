#!/bin/bash

# Script to download data from a Kaggle competition

# Competition slug
COMPETITION_SLUG="soil-classification-part-2"
TARGET_DIR="./data"

echo "Downloading competition data: $COMPETITION_SLUG"
mkdir -p "$TARGET_DIR"
kaggle competitions download -c "$COMPETITION_SLUG" -p "$TARGET_DIR" --unzip

echo "Download complete. Files saved to $TARGET_DIR"
