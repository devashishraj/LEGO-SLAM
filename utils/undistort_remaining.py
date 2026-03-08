import torch
import os
from tqdm import tqdm
import cv2
import numpy as np
import torch.nn.functional as F
import argparse

def undistort_feature_map_torch(feature_map, map1x, map1y):
    """
    Perform undistortion of feature map (C, H, W) using torch grid_sample on GPU.
    """
    device = feature_map.device
    H, W = feature_map.shape[1:]

    # Convert maps to torch and normalize to [-1, 1] coordinate grid
    map1x_t = torch.from_numpy(map1x).to(device)
    map1y_t = torch.from_numpy(map1y).to(device)

    # Normalize pixel coordinates: (0, W-1) → (-1, 1)
    grid_x = 2.0 * (map1x_t / (W - 1)) - 1.0
    grid_y = 2.0 * (map1y_t / (H - 1)) - 1.0
    grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)  # (1, H, W, 2)

    # Add batch dimension to feature map
    feature_map = feature_map.unsqueeze(0)  # (1, C, H, W)

    # Bilinear sampling (same as cv2.INTER_LINEAR)
    undistorted = F.grid_sample(
        feature_map,
        grid,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True
    )

    # Remove batch dimension
    return undistorted.squeeze(0)


def main(args):
    files_all = os.listdir(args.input_path)
    pt_files = sorted([f for f in files_all if f.endswith(".pt")])

    # Load one file to get feature map shape
    test_file = os.path.join(args.input_path, pt_files[0])
    test_semantic_image = torch.load(test_file, map_location='cpu', weights_only=False)

    semantic_width = test_semantic_image.shape[2]
    semantic_height = test_semantic_image.shape[1]

    # Intrinsics
    fx, fy, cx, cy = args.fx, args.fy, args.cx, args.cy
    k1, k2, p1, p2, k3 = args.k1, args.k2, args.p1, args.p2, args.k3

    # Scaling for feature map resolution
    scale_x = float(semantic_width / args.image_width)
    scale_y = float(semantic_height / args.image_height)

    feat_fx = float(fx * scale_x)
    feat_fy = float(fy * scale_y)
    feat_cx = float(cx * scale_x)
    feat_cy = float(cy * scale_y)

    feat_K = np.array([[feat_fx, 0., feat_cx],
                       [0., feat_fy, feat_cy],
                       [0., 0., 1]], dtype=np.float32)

    dist_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float32)

    # Compute undistortion maps
    map1x_feat, map1y_feat = cv2.initUndistortRectifyMap(
        feat_K, dist_coeffs, np.eye(3), feat_K, (int(semantic_width), int(semantic_height)), cv2.CV_32FC1
    )

    # Filter: only process files that still need processing
    files_to_process = []
    print("Checking which files need processing...")
    for pt_file in tqdm(pt_files):
        file_path = os.path.join(args.input_path, pt_file)
        try:
            # Try loading with weights_only=True
            torch.load(file_path, map_location='cpu', weights_only=True)
            # If successful, already undistorted, skip
        except:
            # If failed, needs processing
            files_to_process.append(pt_file)

    print(f"\nFound {len(files_to_process)} files that need processing (out of {len(pt_files)} total)")

    if len(files_to_process) == 0:
        print("All files already processed!")
        return

    # Process only remaining files
    print("\nProcessing remaining files...")
    for pt_file in tqdm(files_to_process):
        file_path = os.path.join(args.input_path, pt_file)
        semantic_feature_raw = torch.load(file_path, map_location='cpu', weights_only=False).float().cuda()

        semantic_feature_undistorted = undistort_feature_map_torch(
            semantic_feature_raw, map1x=map1x_feat, map1y=map1y_feat
        )

        torch.save(semantic_feature_undistorted.half().cpu(), file_path)

    print(f"\n✓ Successfully processed {len(files_to_process)} files!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Undistort only remaining feature maps that need processing.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to directory containing .pt files")
    parser.add_argument("--image_width", type=int, required=True, help="Original image width before scaling")
    parser.add_argument("--image_height", type=int, required=True, help="Original image height before scaling")
    parser.add_argument("--fx", type=float, required=True, help="Focal length in x")
    parser.add_argument("--fy", type=float, required=True, help="Focal length in y")
    parser.add_argument("--cx", type=float, required=True, help="Principal point x")
    parser.add_argument("--cy", type=float, required=True, help="Principal point y")
    parser.add_argument("--k1", type=float, required=True, help="Distortion coefficient k1")
    parser.add_argument("--k2", type=float, required=True, help="Distortion coefficient k2")
    parser.add_argument("--p1", type=float, required=True, help="Distortion coefficient p1")
    parser.add_argument("--p2", type=float, required=True, help="Distortion coefficient p2")
    parser.add_argument("--k3", type=float, required=True, help="Distortion coefficient k3")

    args = parser.parse_args()
    main(args)
