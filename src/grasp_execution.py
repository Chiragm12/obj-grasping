import torch
import numpy as np
from .pybullet_simulation import PyBulletGraspSimulation
from .dcnn_model import GraspDCNN
from .utils import load_config, normalize_depth_for_training, visualize_grasp
import time

class GraspExecutor:
    def __init__(self, model_path, config_path="config/config.yaml"):
        self.config = load_config(config_path)
        
        # Load trained model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = GraspDCNN(config_path).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        print(f"Loaded model from {model_path}")
    
    def predict_grasp(self, depth_image):
        """Predict grasp pose from depth image"""
        # Normalize depth
        normalized_depth, z_offset = normalize_depth_for_training(depth_image)
        
        # Convert to tensor
        depth_tensor = torch.FloatTensor(normalized_depth).unsqueeze(0).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            prediction = self.model(depth_tensor)
            grasp_pose = prediction.cpu().numpy().squeeze()
        
        return grasp_pose
    
    def test_grasping(self, num_tests=10, gui=True):
        """Test grasping on random objects"""
        sim = PyBulletGraspSimulation(gui=gui)
        
        success_count = 0
        results = []
        
        for test_idx in range(num_tests):
            print(f"\nTest {test_idx + 1}/{num_tests}")
            
            # Load random object
            obj_id, obj_name = sim.load_random_object()
            if obj_id is None:
                continue
            
            # Capture depth image
            depth_image, rgb_image = sim.capture_depth_image()
            
            # Predict grasp
            predicted_grasp = self.predict_grasp(depth_image)
            
            # Visualize prediction
            vis_img = visualize_grasp(depth_image, predicted_grasp)
            
            # Execute grasp in simulation
            success = sim.execute_grasp(predicted_grasp, obj_id)
            
            if success:
                success_count += 1
                print(f"✓ Grasp successful on {obj_name}")
            else:
                print(f"✗ Grasp failed on {obj_name}")
            
            results.append({
                'object': obj_name,
                'predicted_grasp': predicted_grasp,
                'success': success,
                'depth_image': depth_image
            })
            
            if gui:
                time.sleep(2)  # Pause for visualization
        
        sim.close()
        
        success_rate = success_count / num_tests * 100
        print(f"\nOverall success rate: {success_rate:.1f}% ({success_count}/{num_tests})")
        
        return results, success_rate

if __name__ == "__main__":
    # Test the trained model
    executor = GraspExecutor('models/grasp_model_final.pth')
    results, success_rate = executor.test_grasping(num_tests=20, gui=True)
