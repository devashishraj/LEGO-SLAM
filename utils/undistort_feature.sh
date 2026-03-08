#!/bin/bash

# Parse command-line arguments
TUM_PATH=${1:-/path/to/TUM}
SCANNET_PATH=${2:-/path/to/Scannet}

echo "TUM_PATH: ${TUM_PATH}"
echo "SCANNET_PATH: ${SCANNET_PATH}"


### TUM
# fr1 - In-place undistortion (overwrite original)
python undistort_feature_img.py --input_path ${TUM_PATH}/rgbd_dataset_freiburg1_desk/rgb_feature_langseg \
                                --output_path ${TUM_PATH}/rgbd_dataset_freiburg1_desk/rgb_feature_langseg \
                                --image_width 640 \
                                --image_height 480 \
                                --fx 517.3 \
                                --fy 516.5 \
                                --cx 318.6 \
                                --cy 255.3 \
                                --k1 0.2624 \
                                --k2 -0.9531 \
                                --p1 -0.0054 \
                                --p2 0.0026 \
                                --k3 1.1633
# fr2 - In-place undistortion (overwrite original)
python undistort_feature_img.py --input_path ${TUM_PATH}/rgbd_dataset_freiburg2_xyz/rgb_feature_langseg \
                                --output_path ${TUM_PATH}/rgbd_dataset_freiburg2_xyz/rgb_feature_langseg \
                                --image_width 640 \
                                --image_height 480 \
                                --fx 520.9 \
                                --fy 521.0 \
                                --cx 325.1 \
                                --cy 249.7 \
                                --k1 0.2312 \
                                --k2 -0.7849 \
                                --p1 -0.0033 \
                                --p2 -0.0001 \
                                --k3 0.9172
# fr3 - In-place undistortion (overwrite original)
python undistort_feature_img.py --input_path ${TUM_PATH}/rgbd_dataset_freiburg3_long_office_household/rgb_feature_langseg \
                                --output_path ${TUM_PATH}/rgbd_dataset_freiburg3_long_office_household/rgb_feature_langseg \
                                --image_width 640 \
                                --image_height 480 \
                                --fx 535.4 \
                                --fy 539.2 \
                                --cx 320.1 \
                                --cy 247.6 \
                                --k1 0.0 \
                                --k2 0.0 \
                                --p1 0.0 \
                                --p2 0.0 \
                                --k3 0.0

### Scannet - In-place undistortion (overwrite original)

for scene in scene0000_00 scene0059_00 scene0106_00 scene0169_00 scene0181_00 scene0207_00
do
    python undistort_feature_img.py --input_path ${SCANNET_PATH}/${scene}/rgb_feature_langseg \
                                --output_path ${SCANNET_PATH}/${scene}/rgb_feature_langseg \
                                --image_width 640 \
                                --image_height 480 \
                                --fx 577.590698 \
                                --fy 578.729797 \
                                --cx 318.905426 \
                                --cy 242.683609 \
                                --k1 0.0 \
                                --k2 0.0 \
                                --p1 0.0 \
                                --p2 0.0 \
                                --k3 1.163314
done
