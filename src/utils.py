import numpy as np
import torch
import yaml
import cv2
from scipy.spatial.transform import Rotation as R

def load_config(config_path="config/config.yaml"):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def euler_to_quaternion(roll, pitch, yaw):
    """Convert Euler angles to quaternion"""
    r = R.from_euler('xyz', [roll, pitch, yaw])
    return r.as_quat()  # [x, y, z, w]

def quaternion_to_euler(quat):
    """Convert quaternion to Euler angles"""
    r = R.from_quat(quat)
    return r.as_euler('xyz')

def process_depth_image(depth_raw, near=0.1, far=2.0):
    """Process raw PyBullet depth buffer to actual depth values"""
    depth = far * near / (far - (far - near) * depth_raw)
    return depth

def normalize_depth_for_training(depth_image, target_distance=0.6):
    """Normalize depth image to target distance"""
    # Find object in image (non-background pixels)
    valid_pixels = depth_image[depth_image < 1.9]  # Exclude far background
    
    if len(valid_pixels) == 0:
        return depth_image, 0
    
    avg_distance = np.mean(valid_pixels)
    offset = target_distance - avg_distance
    
    # Apply normalization
    normalized_depth = depth_image.copy()
    normalized_depth[depth_image < 1.9] += offset
    
    return normalized_depth, offset

def visualize_grasp(depth_image, grasp_pose, save_path=None):
    """Visualize grasp pose on depth image"""
    # Convert depth to displayable image
    depth_vis = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
    
    # Project grasp position to image coordinates
    h, w = depth_image.shape
    u = int(w/2 + grasp_pose[0] * 200)  # Simple projection
    v = int(h/2 + grasp_pose[1] * 200)
    
    # Draw grasp point and orientation
    cv2.circle(depth_vis, (u, v), 5, (0, 255, 0), -1)
    
    # Draw approach direction arrow
    arrow_length = 30
    end_u = int(u + arrow_length * np.cos(grasp_pose[5]))
    end_v = int(v + arrow_length * np.sin(grasp_pose[5]))
    cv2.arrowedLine(depth_vis, (u, v), (end_u, end_v), (255, 0, 0), 2)
    
    if save_path:
        cv2.imwrite(save_path, depth_vis)
    
    return depth_vis
