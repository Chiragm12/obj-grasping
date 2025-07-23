import numpy as np
import os
import pickle
from .pybullet_simulation import PyBulletGraspSimulation
from .utils import load_config, normalize_depth_for_training, visualize_grasp
import matplotlib.pyplot as plt
import pybullet as p
class GraspDatasetGenerator:
    def __init__(self, config_path="config/config.yaml"):
        self.config = load_config(config_path)
        self.data_config = self.config['data_generation']
        
        # Initialize simulation
        self.sim = PyBulletGraspSimulation(gui=False, config_path=config_path)
        
    def normalize_grasp_pose(self, grasp_pose):
        """Normalize grasp pose to reasonable ranges"""
        normalized = grasp_pose.copy()
        
        # Normalize positions to [-0.5, 0.5] range
        normalized[:3] = normalized[:3] / 0.5
        
        # Normalize rotations to [-1, 1] range (from [-π, π])
        normalized[3:] = normalized[3:] / np.pi
        
        return normalized
    
    def denormalize_grasp_pose(self, normalized_pose):
        """Convert normalized pose back to real coordinates"""
        denormalized = normalized_pose.copy()
        
        # Denormalize positions
        denormalized[:3] = denormalized[:3] * 0.5
        
        # Denormalize rotations
        denormalized[3:] = denormalized[3:] * np.pi
        
        return denormalized
    
    def generate_single_sample(self):
        """Generate a single training sample with better quality control"""
        max_attempts = 5
        
        for attempt in range(max_attempts):
            # Load random object
            obj_id, obj_name = self.sim.load_random_object()
            
            if obj_id is None:
                continue
            
            # Capture depth image from random camera position
            camera_distance = np.random.normal(
                self.data_config['camera_distance'],
                self.data_config['distance_variance']
            )
            
            # Add small random camera offset
            camera_pos = [
                np.random.uniform(-0.05, 0.05),
                np.random.uniform(-0.05, 0.05),
                max(0.3, camera_distance)  # Ensure minimum distance
            ]
            
            depth_image, rgb_image = self.sim.capture_depth_image(camera_pos)
            
            # Check if object is visible in image
            if np.sum(depth_image < 1.5) < 100:  # Not enough object pixels
                continue
            
            # Generate better grasp candidates
            grasp_candidates = self.generate_quality_grasps(obj_id, num_candidates=100)
            
            if len(grasp_candidates) == 0:
                continue
            
            # Select best grasp
            best_grasp, quality = self.sim.select_best_grasp(grasp_candidates, obj_id)
            
            if best_grasp is not None and quality > 0.2:  # Higher quality threshold
                # Normalize depth image
                normalized_depth, z_offset = normalize_depth_for_training(depth_image)
                
                # Adjust and normalize grasp
                adjusted_grasp = best_grasp.copy()
                adjusted_grasp[2] += z_offset
                
                # Normalize grasp pose
                normalized_grasp = self.normalize_grasp_pose(adjusted_grasp)
                
                sample = {
                    'depth_image': normalized_depth,
                    'grasp_pose': normalized_grasp,  # Store normalized pose
                    'original_pose': adjusted_grasp,  # Keep original for reference
                    'quality': quality,
                    'object_name': obj_name,
                    'camera_pos': camera_pos
                }
                
                return sample
        
        return None
    
    def generate_quality_grasps(self, object_id, num_candidates=100):
        """Generate higher quality grasp candidates"""
        pos, orn = p.getBasePositionAndOrientation(object_id)
        aabb_min, aabb_max = p.getAABB(object_id)
        
        candidates = []
        object_center = np.array(pos)
        object_size = np.array(aabb_max) - np.array(aabb_min)
        
        for _ in range(num_candidates):
            # Different strategies based on object type
            if object_size[0] < 0.15 and object_size[1] < 0.15:  # Small objects
                # Top-down grasps for small objects
                grasp_x = object_center[0] + np.random.uniform(-0.02, 0.02)
                grasp_y = object_center[1] + np.random.uniform(-0.02, 0.02)
                grasp_z = aabb_max[2] + np.random.uniform(0.03, 0.08)  # Above object
                
                # Prefer vertical approach for small objects
                roll = np.random.uniform(-np.pi/8, np.pi/8)   # ±22.5 degrees
                pitch = np.random.uniform(-np.pi/8, np.pi/8)  # ±22.5 degrees
                yaw = np.random.uniform(-np.pi, np.pi)
            else:
                # Side grasps for larger objects
                offset_range = min(0.04, max(object_size[:2]) * 0.3)
                grasp_x = object_center[0] + np.random.uniform(-offset_range, offset_range)
                grasp_y = object_center[1] + np.random.uniform(-offset_range, offset_range)
                grasp_z = np.random.uniform(aabb_min[2] + 0.02, aabb_max[2] + 0.05)
                
                roll = np.random.uniform(-np.pi/4, np.pi/4)
                pitch = np.random.uniform(-np.pi/4, np.pi/4)
                yaw = np.random.uniform(-np.pi, np.pi)
            
            grasp_pose = np.array([grasp_x, grasp_y, grasp_z, roll, pitch, yaw])
            
            # Feasibility check
            if (aabb_min[0] - 0.05 <= grasp_x <= aabb_max[0] + 0.05 and
                aabb_min[1] - 0.05 <= grasp_y <= aabb_max[1] + 0.05 and
                grasp_z >= aabb_min[2]):
                candidates.append(grasp_pose)
        
        return candidates

    
    def generate_dataset(self, total_samples=2000, output_dir="data/generated_training"):
        """Generate complete training dataset with quality control"""
        os.makedirs(output_dir, exist_ok=True)
        
        samples = []
        successful_samples = 0
        attempts = 0
        
        print(f"Generating {total_samples} high-quality training samples...")
        
        while successful_samples < total_samples and attempts < total_samples * 5:
            attempts += 1
            
            sample = self.generate_single_sample()
            
            if sample is not None:
                samples.append(sample)
                successful_samples += 1
                
                # Save sample visualization every 200 samples
                if successful_samples % 200 == 0:
                    vis_path = os.path.join(output_dir, f"sample_{successful_samples}.png")
                    # Use original pose for visualization
                    visualize_grasp(
                        sample['depth_image'], 
                        sample['original_pose'], 
                        save_path=vis_path
                    )
                    
                    print(f"Generated {successful_samples}/{total_samples} samples "
                          f"(Success rate: {successful_samples/attempts*100:.1f}%)")
        
        # Save dataset
        dataset_path = os.path.join(output_dir, 'training_dataset.pkl')
        with open(dataset_path, 'wb') as f:
            pickle.dump(samples, f)
        
        # Print statistics
        qualities = [s['quality'] for s in samples]
        print(f"\nDataset generation complete!")
        print(f"Total samples: {len(samples)}")
        print(f"Success rate: {len(samples)/attempts*100:.1f}%")
        print(f"Average quality: {np.mean(qualities):.3f}")
        print(f"Quality range: {np.min(qualities):.3f} - {np.max(qualities):.3f}")
        
        return samples
    
    def close(self):
        """Close simulation"""
        self.sim.close()
