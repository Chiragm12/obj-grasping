from src.grasp_execution import GraspExecutor
from src.pybullet_simulation import PyBulletGraspSimulation
import time
import os

def test_bowl_object():
    print("=== Testing Bowl Object Grasping ===")
    
    # Check if bowl.obj exists
    if not os.path.exists("custom_objects/bowl.obj"):
        print("❌ bowl.obj not found in custom_objects/ directory")
        return
    
    # Create grasp executor
    executor = GraspExecutor('models/grasp_model_final.pth')
    
    # Create simulation
    sim = PyBulletGraspSimulation(gui=True)
    
    # First, find the best scale for the bowl
    print("Finding optimal scale for bowl...")
    best_scale = sim.test_multiple_scales('bowl')
    
    if best_scale is None:
        print("❌ Could not find working scale for bowl")
        sim.close()
        return
    
    print(f"Using scale: {best_scale}")
    
    # Test grasping multiple times
    success_count = 0
    total_tests = 5
    
    for test_num in range(total_tests):
        print(f"\n--- Test {test_num + 1}/{total_tests} ---")
        
        # Load bowl object
        obj_id, obj_name = sim.load_custom_object('bowl', scale=best_scale)
        
        if obj_id is None:
            print("Failed to load bowl")
            continue
        
        # Capture depth image
        depth_image, rgb_image = sim.capture_depth_image()
        
        # Predict grasp
        predicted_grasp = executor.predict_grasp(depth_image)
        
        # Execute grasp
        success = sim.execute_grasp(predicted_grasp, obj_id)
        
        if success:
            success_count += 1
            print("✅ Grasp SUCCESSFUL")
        else:
            print("❌ Grasp failed")
        
        # Show grasp details
        pos = predicted_grasp[:3]
        rot = predicted_grasp[3:]
        print(f"Predicted position: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
        print(f"Predicted rotation: [{rot[0]:.3f}, {rot[1]:.3f}, {rot[2]:.3f}]")
        
        time.sleep(2)  # Pause between tests
    
    # Results
    success_rate = success_count / total_tests * 100
    print(f"\n=== RESULTS ===")
    print(f"Success Rate: {success_rate:.1f}% ({success_count}/{total_tests})")
    print(f"bowl Scale Used: {best_scale}")
    
    sim.close()

if __name__ == "__main__":
    test_bowl_object()
