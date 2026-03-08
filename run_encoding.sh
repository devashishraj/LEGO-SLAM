#!/bin/bash

# Usage:
#   bash run_encoding.sh --dataset_path <path> --scenes "<scene1> <scene2> ..." --rgb_dir <folder_name>
#
# Examples:
#   bash run_encoding.sh --dataset_path /path/to/Replica --scenes "office0 office1 room0" --rgb_dir frame
#   bash run_encoding.sh --dataset_path /path/to/Scannet --scenes "scene0000_00 scene0059_00" --rgb_dir color
#   bash run_encoding.sh --dataset_path /path/to/TUM --scenes "rgbd_dataset_freiburg2_xyz" --rgb_dir rgb

DATASET_PATH=""
SCENES=""
RGB_DIR="images"

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset_path) DATASET_PATH="$2"; shift 2 ;;
        --scenes) SCENES="$2"; shift 2 ;;
        --rgb_dir) RGB_DIR="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ -z "$DATASET_PATH" ] || [ -z "$SCENES" ]; then
    echo "Error: --dataset_path and --scenes are required."
    echo "Usage: bash run_encoding.sh --dataset_path <path> --scenes \"<scene1> <scene2>\" --rgb_dir <folder_name>"
    exit 1
fi

for SCENE in $SCENES; do
    DATA_DIR=${DATASET_PATH}/${SCENE}
    echo "Encoding: ${DATA_DIR}"
    python -u Lseg/encode_images.py \
        --backbone clip_vitl16_384 \
        --weights Lseg/demo_e200.ckpt \
        --widehead --no-scaleinv \
        --outdir ${DATA_DIR}/rgb_feature_langseg \
        --test-rgb-dir ${DATA_DIR}/${RGB_DIR} \
        --workers 0
done
