import os
import torch
import torch.nn.functional as F
import copy
import sys
import numpy as np
import rerun as rr
import pygicp
import pickle
sys.path.append(os.path.dirname(__file__))
from arguments import SLAMParameters
from utils.PGO import *
from dataclasses import dataclass

@dataclass
class MatchingResult:
    score: float
    target_kf_id: int
    relative_pose: np.array

class LoopManager(SLAMParameters):
    def __init__(self, slam):
        # Store reference to mapper for accessing histograms (if language detection enabled)
        self.mapper = slam
        
        self.downsample_idxs = slam.downsample_idxs
        self.x_pre = slam.x_pre
        self.y_pre = slam.y_pre
        self.downsample_idxs_sparse = slam.downsample_idxs_sparse
        self.x_pre_sparse = slam.x_pre_sparse
        self.y_pre_sparse = slam.y_pre_sparse
        self.depth_trunc = slam.depth_trunc
        self.depth_scale = 1.0
        self.rerun_viewer = slam.rerun_viewer
        
        self.W = slam.W
        self.H = slam.H
        self.fx = slam.fx
        self.fy = slam.fy
        self.cx = slam.cx
        self.cy = slam.cy
        self.loop_closing_start = slam.loop_closing_start
        
        self.loop_candidates = []
        self.collected_loops = []   # value1, value2, relative_T
        self.num_updated_loops = 0
        self.loop_closing_trigger = False
        
        self.loopclosing_global_correspondence_distance = slam.loopclosing_global_correspondence_distance
        self.loopclosing_local_correspondence_distance = slam.loopclosing_local_correspondence_distance
        self.loop_constraint_noise = slam.loop_constraint_noise
        
        self.submap_margin = 5
        
        self.running_loop_closing = False
        
        self.kf_poses_before_pgo = []
        self.kf_poses_after_pgo = []

        print("LoopManager: Language-based loop detection enabled")
        # Load codebook for histogram generation (independent of mapper)
        self.codebook_path = "saved/language_codebook_64.pkl"
        self.vocabulary = None
        self.vocabulary_gpu = None
        self.num_clusters = 64
        self.load_codebook()
        
        if self.rerun_viewer:
            rr.init("3dgsviewer")
            rr.connect()
    
    def depth2pc(self, depth_img):
        '''
        input: depth img (tensor)
        output: point cloud (numpy)
        '''

        z_values = depth_img.flatten()[self.downsample_idxs]/self.depth_scale
        zero_filter = torch.where(z_values!=0)
        filter = torch.where(z_values[zero_filter]<=self.depth_trunc)
        # Trackable gaussians (will be used in tracking)
        z_values = z_values[zero_filter]
        x = self.x_pre[zero_filter] * z_values
        y = self.y_pre[zero_filter] * z_values
        points = torch.stack([x,y,z_values], dim=-1)
        
        z_values_sparse = depth_img.flatten()[self.downsample_idxs_sparse]/self.depth_scale
        zero_filter_sparse = z_values_sparse!=0
        z_filter_sparse = z_values_sparse<=self.depth_trunc
        filter_sparse = zero_filter_sparse & z_filter_sparse
        z_values_sparse = z_values_sparse[filter_sparse]
        x_sparse = self.x_pre_sparse[filter_sparse] * z_values_sparse
        y_sparse = self.y_pre_sparse[filter_sparse] * z_values_sparse
        points_sparse = torch.stack([x_sparse,y_sparse,z_values_sparse], dim=-1)
        
        return points.numpy(), points_sparse.numpy()
    
    def input_gs_map(self, points, rots, scales, trackable_mask, kf_idxs):
        self.map_points = copy.deepcopy(points)
        self.map_rots = copy.deepcopy(rots)
        self.map_scales = copy.deepcopy(scales)
        self.map_trackable_mask = copy.deepcopy(trackable_mask)
        self.map_kf_idxs = copy.deepcopy(kf_idxs)
    
    def run(self):
        pass
        # while True:
        #     time.sleep(1e-3)

        #     if self.loop_closing_trigger:
        #         self.loop_detection_naive()
            
        #         self.loop_closing_trigger = False
    
    def load_codebook(self):
        """Load language codebook for histogram generation"""
        if not os.path.exists(self.codebook_path):
            print(f"Warning: Codebook file not found: {self.codebook_path}")
            return
        
        try:
            with open(self.codebook_path, 'rb') as f:
                codebook_data = pickle.load(f)
            
            self.vocabulary = codebook_data['vocabulary']
            self.num_clusters = codebook_data['num_clusters']
            
            print(f"LoopManager: Codebook loaded: {self.num_clusters} clusters x {self.vocabulary.shape[1]} dimensions")
        except Exception as e:
            print(f"LoopManager: Error loading codebook: {e}")
            self.vocabulary = None
    
    def assign_pixels_to_codebook(self, feature_tensor):
        """Assign each pixel to the most similar codebook entry using cosine similarity (GPU optimized)"""
        if self.vocabulary is None:
            return None
            
        # Convert Float16 → Float32 if needed
        if feature_tensor.dtype == torch.float16:
            feature_tensor = feature_tensor.float()
        
        # Reshape: [512, H, W] → [H*W, 512] (keep on GPU)
        C, H, W = feature_tensor.shape
        pixels_gpu = feature_tensor.permute(1, 2, 0).reshape(-1, C)  # [H*W, 512], stays on GPU
        
        # Move vocabulary to GPU (cache it for efficiency)
        if self.vocabulary_gpu is None:
            self.vocabulary_gpu = torch.from_numpy(self.vocabulary).float().cuda()  # [64, 512]
        
        # Calculate cosine similarity on GPU (much faster!)
        pixels_normalized = F.normalize(pixels_gpu, dim=1)  # [H*W, 512]
        vocab_normalized = F.normalize(self.vocabulary_gpu, dim=1)  # [64, 512]
        
        similarities = torch.mm(pixels_normalized, vocab_normalized.t())  # [H*W, 64]
        
        # Find most similar codebook (highest similarity)
        assignments = torch.argmax(similarities, dim=1)  # [H*W]
        assignments = assignments.reshape(H, W).cpu().numpy()  # Only final result goes to CPU
        
        return assignments
    
    def create_histogram(self, assignments):
        """Create histogram from assignments"""
        if assignments is None:
            return None
            
        # Calculate histogram
        histogram, _ = np.histogram(assignments.flatten(), 
                                 bins=self.num_clusters, 
                                 range=(0, self.num_clusters))
        
        # Normalize (convert to probability)
        histogram = histogram.astype(np.float32)
        if histogram.sum() > 0:
            histogram = histogram / histogram.sum()
        
        return histogram
    
    def generate_current_histogram(self, semantic_feature):
        """Generate histogram for current semantic feature"""
        if self.vocabulary is None:
            return None
            
        try:
            # Assign pixels to codebook
            assignments = self.assign_pixels_to_codebook(semantic_feature)
            if assignments is None:
                return None
            
            # Create histogram
            histogram = self.create_histogram(assignments)
            return histogram
            
        except Exception as e:
            print(f"LoopManager: Error generating current histogram: {e}")
            return None
    
    def insert_new_kf(self, new_kf_pose, kf_depth, kf_idx, all_kf_poses, tracking_kf_idxs):

        # point cloud of inserted keyframe
        self.new_kf_pose = torch.from_numpy(new_kf_pose)
        self.new_kf_pc, self.new_kf_pc_sparse = self.depth2pc(kf_depth.detach().cpu())
        self.new_kf_idx = kf_idx
        self.all_kf_poses = all_kf_poses
        self.tracking_kf_idxs = tracking_kf_idxs

        # remove recent N keyframes
        self.tracking_kf_idxs_cropped = tracking_kf_idxs[:-self.loop_closing_start]
        
        self.loop_closing_trigger = True
    
    def loop_detection_naive(self):
        self.loop_detection_language_feature()
    
    def loop_detection_language_feature(self):
        """Language feature-based loop detection using histograms (Optimized)"""
        # Get current histogram from mapper
        if not hasattr(self.mapper, 'histograms') or self.mapper.histograms is None:
            print("No histograms available from mapper")
            return
        
        current_histogram = self.mapper.histograms.get(self.new_kf_idx, None)
        if current_histogram is None:
            print(f"No histogram found for keyframe {self.new_kf_idx}")
            return
            
        new_kf_pose_c2w = self.new_kf_pose
        self.loop_candidates.clear()
        
        # Pre-convert and normalize current histogram (outside the loop - major optimization!)
        current_hist_tensor = torch.from_numpy(current_histogram).float().cuda()
        current_norm = F.normalize(current_hist_tensor, dim=0)
        
        new_loop_candidates = []
        scores = []
        
        # Extract position once for distance calculations
        new_kf_position = new_kf_pose_c2w[:3, 3]
        
        # find loop candidates
        for idx_in_tracking_kfs, kf_idx in enumerate(self.tracking_kf_idxs_cropped):
            kf_pose = self.all_kf_poses[kf_idx]
            
            # Optimized pose distance check (faster than linalg.norm)
            pose_diff = new_kf_position - kf_pose[:3, 3]
            pose_dist_sq = torch.sum(pose_diff * pose_diff)
            
            if pose_dist_sq > 1.0:  # Compare squared distance (faster)
                continue
            
            # Check if histogram exists for candidate keyframe
            if kf_idx not in self.mapper.histograms:
                continue
            
            # Get candidate histogram and convert to tensor (only when needed)
            candidate_histogram = self.mapper.histograms[kf_idx]
            candidate_hist_tensor = torch.from_numpy(candidate_histogram).float().cuda()
            candidate_norm = F.normalize(candidate_hist_tensor, dim=0)
            
            # Calculate cosine similarity (GPU accelerated)
            similarity = torch.dot(current_norm, candidate_norm).item()
            
            # Use similarity threshold for detection (optimized threshold)
            if similarity > 0.7:
                new_loop_candidates.append([idx_in_tracking_kfs, self.all_kf_poses[kf_idx]])
                scores.append(similarity)
        
        if len(scores) == 0:
            return
            
        # Use PyTorch for top-k selection 
        scores_tensor = torch.tensor(scores, device='cuda')
        topk_values, topk_indices = torch.topk(scores_tensor, min(2, len(scores)), largest=True)
        # topk_values, topk_indices = torch.topk(scores_tensor, min(3, len(scores)), largest=True)
        
        # Extract top candidates
        for i in topk_indices.cpu():
            if i < len(new_loop_candidates):  # Safety check
                self.loop_candidates.append(new_loop_candidates[i])
        
        # Optional debug info (only when loops found)
    def get_projection_filtered_score(self,
                                    source_pc,
                                    target_pc,
                                    pose_T_target_source,
                                    gicp_distances_sq,
                                    inlier_threshold=0.05):
        """
        Filter GICP distance results using a projection mask and compute a score.

        1. Project target PC onto source camera plane to create an overlap 2D mask.
        2. Project source PC and filter points using the overlap mask.
        3. Compute overlap score and RMSE using only the filtered GICP distances.

        Args:
            source_pc (np.array): Source point cloud.
            target_pc (np.array): Target point cloud (submap).
            pose_T_target_source (np.array): GICP transform matrix (Source -> Target).
            gicp_distances_sq (np.array): Squared distances from reg.get_source_correspondence().
            inlier_threshold (float): Distance threshold for inliers (meters).

        Returns:
            tuple: (overlap_score, rmse)
        """
        # 1. Project target PC to create a 2D occupancy mask
        T_source_target = np.linalg.inv(pose_T_target_source)
        R_st, t_st = T_source_target[:3, :3], T_source_target[:3, 3]
        target_in_source_frame = (R_st @ target_pc.T).T + t_st

        z_t = target_in_source_frame[:, 2]
        u_t = np.round((self.fx * target_in_source_frame[:, 0] / z_t) + self.cx).astype(int)
        v_t = np.round((self.fy * target_in_source_frame[:, 1] / z_t) + self.cy).astype(int)

        valid_proj_t = (u_t >= 0) & (u_t < self.W) & (v_t >= 0) & (v_t < self.H) & (z_t > 0) & (z_t < 5.0)
        valid_u_t = u_t[valid_proj_t]
        valid_v_t = v_t[valid_proj_t]

        # 2D mask: True where a target point was projected
        projection_mask = np.full((self.H, self.W), False, dtype=bool)
        projection_mask[valid_v_t, valid_u_t] = True

        # 2. Project source PC and filter using the mask
        z_s = source_pc[:, 2]
        u_s = np.round((self.fx * source_pc[:, 0] / z_s) + self.cx).astype(int)
        v_s = np.round((self.fy * source_pc[:, 1] / z_s) + self.cy).astype(int)
        
        valid_proj_s = (u_s >= 0) & (u_s < self.W) & (v_s >= 0) & (v_s < self.H) & (z_s > 0)
        source_filter = np.full(len(source_pc), False, dtype=bool)
        valid_indices_s = np.where(valid_proj_s)[0]
        
        # Check mask only for valid projected indices
        projected_pixels_are_valid = projection_mask[v_s[valid_indices_s], u_s[valid_indices_s]]
        source_filter[valid_indices_s[projected_pixels_are_valid]] = True

        # print(source_filter.sum())
        if source_filter.sum() < 100:   # 200
            return 0.0, np.inf

        # Dynamic inlier threshold based on depth
        filtered_distances_sq = gicp_distances_sq[source_filter]
        filtered_source_depths = source_pc[source_filter, 2]
        
        min_inlier_threshold = 0.005
        max_inlier_threshold = 0.1
        depth_factor = (max_inlier_threshold - min_inlier_threshold)/5.
        
        # Per-point dynamic threshold (linear model)
        dynamic_thresholds = min_inlier_threshold + depth_factor * filtered_source_depths
        
        # Clamp thresholds
        dynamic_thresholds = np.clip(dynamic_thresholds, 
                                    min_inlier_threshold, 
                                    0.08)

        inlier_mask = filtered_distances_sq < (dynamic_thresholds**2)
        
        num_inliers = inlier_mask.sum()
        num_filtered_points = len(filtered_distances_sq)

        overlap_score = num_inliers / num_filtered_points

        # 3. Compute score from filtered GICP distances
        
        if num_inliers == 0:
            rmse = np.inf
        else:
            rmse = np.sqrt(filtered_distances_sq[inlier_mask].mean())
            
        return overlap_score, rmse
    
    def evaluate_registration_spatially(self,
                                        source_pc,
                                        target_pc,
                                        pose_T_target_source,
                                        gicp_distances_sq,
                                        inlier_threshold=0.05,
                                        downscale_factor=1.0/2.0,
                                        grid_size=(5, 5)):
        """
        Evaluate registration spatially by dividing the image plane into a grid.
        """
        # 0. Compute downscaled resolution and camera parameters
        scale = downscale_factor
        W_scaled = int(self.W * scale)
        H_scaled = int(self.H * scale)
        fx_scaled = self.fx * scale
        fy_scaled = self.fy * scale
        cx_scaled = self.cx * scale
        cy_scaled = self.cy * scale

        # 1. Project target PC to create 2D mask
        T_source_target = np.linalg.inv(pose_T_target_source)
        R_st, t_st = T_source_target[:3, :3], T_source_target[:3, 3]
        target_in_source_frame = (R_st @ target_pc.T).T + t_st
        z_t = target_in_source_frame[:, 2]
        u_t = np.round((fx_scaled * target_in_source_frame[:, 0] / z_t) + cx_scaled).astype(int)
        v_t = np.round((fy_scaled * target_in_source_frame[:, 1] / z_t) + cy_scaled).astype(int)
        valid_proj_t = (u_t >= 0) & (u_t < W_scaled) & (v_t >= 0) & (v_t < H_scaled) & (z_t > 0) & (z_t < 5.0)
        valid_u_t = u_t[valid_proj_t]
        valid_v_t = v_t[valid_proj_t]
        projection_mask = np.full((H_scaled, W_scaled), False, dtype=bool)
        projection_mask[valid_v_t, valid_u_t] = True

        # 2. Project source PC and filter using the mask
        z_s = source_pc[:, 2]
        u_s = np.round((fx_scaled * source_pc[:, 0] / z_s) + cx_scaled).astype(int)
        v_s = np.round((fy_scaled * source_pc[:, 1] / z_s) + cy_scaled).astype(int)
        valid_proj_s = (u_s >= 0) & (u_s < W_scaled) & (v_s >= 0) & (v_s < H_scaled) & (z_s > 0)
        source_filter = np.full(len(source_pc), False, dtype=bool)
        valid_indices_s = np.where(valid_proj_s)[0]
        projected_pixels_are_valid = projection_mask[v_s[valid_indices_s], u_s[valid_indices_s]]
        source_filter[valid_indices_s[projected_pixels_are_valid]] = True

        if source_filter.sum() == 0:
            return 0, np.inf

        # 3. Assign filtered points to grid cells and compute per-cell scores
        grid_w, grid_h = grid_size
        cell_width = W_scaled / grid_w
        cell_height = H_scaled / grid_h
        
        # Source points to evaluate
        filtered_indices = np.where(source_filter)[0]
        u_s_filtered = u_s[filtered_indices]
        v_s_filtered = v_s[filtered_indices]
        distances_sq_filtered = gicp_distances_sq[filtered_indices]
        
        # Compute grid cell index for each point
        grid_x_indices = (u_s_filtered / cell_width).astype(int)
        grid_y_indices = (v_s_filtered / cell_height).astype(int)

        num_successful_cells = 0
        inlier_threshold_sq = inlier_threshold**2
        
        # Collect inlier distances for overall RMSE
        total_inlier_distances_sq = []

        for i in range(grid_w):
            for j in range(grid_h):
                # Points belonging to cell (i, j)
                in_cell_mask = (grid_x_indices == i) & (grid_y_indices == j)
                
                num_points_in_cell = in_cell_mask.sum()
                
                # Skip cells with too few points
                if num_points_in_cell < 10:
                    continue
                
                cell_distances_sq = distances_sq_filtered[in_cell_mask]
                
                # Compute inliers within this cell
                cell_inlier_mask = cell_distances_sq < inlier_threshold_sq
                num_inliers_in_cell = cell_inlier_mask.sum()
                
                inlier_ratio_in_cell = num_inliers_in_cell / num_points_in_cell
                
                # Count cell as successful if inlier ratio exceeds threshold
                if inlier_ratio_in_cell > 0.5:
                    num_successful_cells += 1

                # Collect inlier distances for overall RMSE
                total_inlier_distances_sq.append(cell_distances_sq[cell_inlier_mask])

        if not total_inlier_distances_sq:
            return 0, np.inf

        # Overall RMSE across all cells
        overall_rmse = np.sqrt(np.concatenate(total_inlier_distances_sq).mean()) if total_inlier_distances_sq else np.inf

        return num_successful_cells, overall_rmse

    def calculate_loop_constraints(self):
        
        reg = pygicp.FastGICP()
        reg.set_max_correspondence_distance(0.3)
        reg.set_max_knn_distance(9999.0)
        reg.set_num_threads(16)
        reg.set_edge_weight(-1.0)   # deactivate edge matching
        reg.set_sparse_weight(0.0)
        
        if len(self.loop_candidates) == 0:
            return
        
        reg.set_input_source(self.new_kf_pc)
        # reg.set_input_source_sparse(self.new_kf_pc_sparse)
        num_trackable_points = self.new_kf_pc.shape[0]
        input_filter = np.arange(1, num_trackable_points + 1, dtype=np.int32)
        reg.set_source_filter(num_trackable_points, input_filter)
        pseudo_filter = np.zeros(self.new_kf_pc.shape[0], dtype=np.int32)
        reg.set_source_edge_invalid_mask(pseudo_filter)
        
        best_set = None
        best_score = -1
        
        for idx_in_tracking_kfs, kf_pose in self.loop_candidates:
            # make submap
            
            # crop_start_idx_ = max(0, idx_in_tracking_kfs - self.submap_margin)
            # crop_start_idx = self.tracking_kf_idxs[crop_start_idx_]
            # crop_end_idx_ = min(len(self.tracking_kf_idxs), idx_in_tracking_kfs + self.submap_margin)
            # crop_end_idx = self.tracking_kf_idxs[crop_end_idx_]
            # submap_filter = torch.logical_and( self.map_trackable_mask, torch.logical_and(self.map_kf_idxs.squeeze(-1) > crop_start_idx, self.map_kf_idxs.squeeze(-1) < crop_end_idx))
            
            # reg.set_max_correspondence_distance(0.5)
            reg.set_max_correspondence_distance(self.loopclosing_global_correspondence_distance)
            reg.set_edge_weight(-1.0)   # deactivate edge matching
            reg.set_sparse_weight(0.0)
            reg.set_max_iteration(40)
            
            mid_idx = self.tracking_kf_idxs[idx_in_tracking_kfs]
            crop_start_idx = max(0, mid_idx-self.submap_margin)
            crop_end_idx = mid_idx + self.submap_margin
            submap_filter = torch.logical_and(self.map_kf_idxs.squeeze(-1) > crop_start_idx, self.map_kf_idxs.squeeze(-1) < crop_end_idx)
            
            submap_points = self.map_points[submap_filter].numpy()
            submap_rots = self.map_rots[submap_filter].numpy()
            submap_scales = self.map_scales[submap_filter].numpy()
            
            # rr.log(
            #     "target_submap",
            #     rr.Points3D(
            #         submap_points
            #     )
            # )
            
            # set submap as target
            reg.set_input_target(submap_points)
            reg.set_target_covariances_fromqs(submap_rots.flatten(), submap_scales.flatten())
            
            
            # match to the submap
            # initial_pose = np.linalg.inv(self.new_kf_pose)
            initial_pose = self.new_kf_pose
            result_T_coarse = reg.align(initial_pose)
            
            is_converged = reg.is_converged()
            if not is_converged:
                # print("coarse gicp not converged, break!")
                continue
            
            reg.set_max_correspondence_distance(self.loopclosing_local_correspondence_distance)   # scannet: 0.05 / tum: 0.03
            reg.set_sparse_weight(0.0)
            reg.set_max_iteration(40)
            
            result_T_final = reg.align(result_T_coarse)
            
            R = result_T_final[:3,:3]
            t = result_T_final[:3,3]
            
            transformed_source = (R @ self.new_kf_pc.T).T + t
            
            target_corres, distances_sq = reg.get_source_correspondence()
            
            overlap_ratio, rmse = self.get_projection_filtered_score(
                source_pc=self.new_kf_pc,
                target_pc=submap_points,
                pose_T_target_source=result_T_final,
                gicp_distances_sq=distances_sq,
                inlier_threshold=1e-3
            )
            
            # constant threshold
            # len_corres = len(np.where(distances_sq < 1e-3)[0])
            # overlap_ratio = len_corres / distances_sq.shape[0]
            
            # overlap_ratio = self.get_overlap_score_cropped(self.new_kf_pc, distances_sq)
            
            # print(overlap_ratio, overlap_score, rmse)
            
            # print(distances.shape, self.new_kf_pc.shape)    # same
            
            # if self.rerun_viewer:
            #     rr.log(
            #         "aligned_source",
            #         rr.Points3D(
            #             transformed_source,
            #             labels=f"score: {overlap_ratio:.3f}/{rmse:.3f}"
            #             # labels=f"score: {num_successful_cells:.3f}/{overall_rmse:.3f}"
            #         )
            #     )
            
            # if overlap_ratio > 0.7:
            #     target_kf_idx = self.tracking_kf_idxs[idx_in_tracking_kfs]
            #     relative_pose = np.linalg.inv(kf_pose) @ result_T_final
            #     self.collected_loops.append([target_kf_idx, self.new_kf_idx, relative_pose])
            
            if overlap_ratio > 0.9 and overlap_ratio > best_score:
                best_score = overlap_ratio
                relative_pose = np.linalg.inv(kf_pose) @ result_T_final
                target_kf_idx = self.tracking_kf_idxs[idx_in_tracking_kfs]
                best_set = [target_kf_idx, self.new_kf_idx, relative_pose]
        
        if best_set != None:
            self.collected_loops.append(best_set)
        
        del reg
        self.loop_candidates.clear()
    
    def get_overlap_score_weighted(self):
        
        pass
    
    def get_overlap_score_cropped(self, raw_pc, distances):
        z_filter = raw_pc[:,2] < 3.5
        inlier_filter = distances < 1e-3    # 1e-3
        
        cropped_inlier_filter = z_filter & inlier_filter
        
        cropped_overlap_ratio = cropped_inlier_filter.sum() / z_filter.sum()
        
        return cropped_overlap_ratio
    
    def pgo_update(self):
        PGM = PoseGraphManager(self.loop_constraint_noise)
        
        # check if new loop collected
        if self.num_updated_loops >= len(self.collected_loops):
            return False
        
        # tracking + mapping keyframes
        for i, kf_pose in enumerate(self.all_kf_poses):
            if i == 0:
                PGM.addPriorFactor(0, kf_pose)
            else:
                prev_kf_pose = self.all_kf_poses[i-1]
                relative_pose = np.linalg.inv(prev_kf_pose) @ kf_pose
                PGM.addOdometryFactor(i-1, i, relative_T=relative_pose, initial_T=kf_pose)
        
        # tracking keyframes
        for idx in range(len(self.tracking_kf_idxs)-1):
            kf_idx = self.tracking_kf_idxs[idx]
            kf_idx_next = self.tracking_kf_idxs[idx+1]
            
            pose = self.all_kf_poses[kf_idx]
            pose_next = self.all_kf_poses[kf_idx_next]
            
            relative_pose = np.linalg.inv(pose) @ pose_next
            PGM.addOdometryFactor_re(kf_idx, kf_idx_next, relative_T=relative_pose)
        
        # loop constraints
        for target_idx, source_idx, relative_pose in self.collected_loops:
            PGM.addLoopFactor(target_idx, source_idx, relative_T=relative_pose)
        
        # print("[loop closing] num_loop_constraints: ", len(self.collected_loops))
        
        # pgo update
        PGM.optimizePoseGraph()
        
        optimized_poses = PGM.getValues(len(self.all_kf_poses))
        
        # vis
        self.kf_poses_before_pgo = []
        self.kf_poses_after_pgo = []
        before_pgo_vis = []
        after_pgo_vis = []
        for i, kf_pose in enumerate(self.all_kf_poses):
            
            self.kf_poses_before_pgo.append(kf_pose)
            self.kf_poses_after_pgo.append(optimized_poses[i])
            before_pgo_vis.append(kf_pose[:3,3])
            after_pgo_vis.append(optimized_poses[i][:3,3])
        
        # if self.rerun_viewer:
        #     rr.log(
        #         f"before_pgo",
        #         rr.Points3D(
        #             np.array(before_pgo_vis),
        #             colors=[0,0,255]),
        #     )
        #
        #     rr.log(
        #         f"after_pgo",
        #         rr.Points3D(np.array(after_pgo_vis),
        #                     colors=[255,0,0])
        #     )
        
        self.num_updated_loops = len(self.collected_loops)

        return True