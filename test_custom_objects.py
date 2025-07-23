from src.grasp_execution import GraspExecutor
from src.pybullet_simulation import PyBulletGraspSimulation
from src.utils import visualize_grasp
import time
import numpy as np
import os

class CustomObjectTester:
    def __init__(self, model_path='models/grasp_model_final.pth'):
        self.executor = GraspExecutor(model_path)
        print(f"Loaded grasp model: {model_path}")
        
    def test_single_custom_object(self, obj_name, scale=1.0, num_tests=5, gui=True):
        """Test model on a single custom .obj file"""
        
        print(f"\n{'='*60}")
        print(f"Testing Custom Object: {obj_name.upper()}")
        print(f"Scale: {scale}x")
        print(f"{'='*60}")
        
        sim = PyBulletGraspSimulation(gui=gui)
        
        results = []
        successful_grasps = []
        failed_grasps = []
        
        for test_num in range(num_tests):
            print(f"\nTest {test_num + 1}/{num_tests}")
            
            # Load the custom object
            obj_id, loaded_name = sim.load_custom_object(obj_name, scale=scale)
            
            if obj_id is None:
                print(f"  ❌ Failed to load object {obj_name}")
                continue
            
            # Capture depth image
            depth_image, rgb_image = sim.capture_depth_image()
            
            # Check if object is visible in depth image
            object_pixels = np.sum(depth_image < 1.5)
            if object_pixels < 50:
                print(f"  ⚠️  Object barely visible in depth image ({object_pixels} pixels)")
            
            # Predict grasp
            predicted_grasp = self.executor.predict_grasp(depth_image)
            
            # Execute grasp in simulation
            success = sim.execute_grasp(predicted_grasp, obj_id)
            
            # Store result
            result = {
                'test_id': test_num + 1,
                'object_name': obj_name,
                'loaded_name': loaded_name,
                'scale': scale,
                'predicted_grasp': predicted_grasp,
                'success': success,
                'object_pixels': object_pixels
            }
            
            results.append(result)
            
            if success:
                successful_grasps.append(predicted_grasp)
                print(f"  ✅ GRASP SUCCESSFUL")
            else:
                failed_grasps.append(predicted_grasp)
                print(f"  ❌ Grasp failed")
            
            # Print grasp details
            pos = predicted_grasp[:3]
            rot = predicted_grasp[3:]
            print(f"     Position: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
            print(f"     Rotation: [{rot[0]:.3f}, {rot[1]:.3f}, {rot[2]:.3f}]")
            
            if gui:
                time.sleep(2)  # Pause for visualization
        
        sim.close()
        
        # Calculate statistics
        success_count = len(successful_grasps)
        success_rate = success_count / len(results) * 100 if results else 0
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"RESULTS SUMMARY FOR {obj_name.upper()}")
        print(f"{'='*60}")
        print(f"Success Rate: {success_rate:.1f}% ({success_count}/{len(results)})")
        print(f"Scale Used: {scale}x")
        
        if successful_grasps:
            print(f"\nSuccessful Grasp Analysis:")
            avg_pos = np.mean([g[:3] for g in successful_grasps], axis=0)
            avg_rot = np.mean([g[3:] for g in successful_grasps], axis=0)
            print(f"  Average Position: [{avg_pos[0]:.3f}, {avg_pos[1]:.3f}, {avg_pos[2]:.3f}]")
            print(f"  Average Rotation: [{avg_rot[0]:.3f}, {avg_rot[1]:.3f}, {avg_rot[2]:.3f}]")
        
        return {
            'object_name': obj_name,
            'scale': scale,
            'results': results,
            'success_rate': success_rate,
            'success_count': success_count,
            'total_tests': len(results),
            'successful_grasps': successful_grasps,
            'failed_grasps': failed_grasps
        }
    
    def test_multiple_custom_objects(self, object_configs, gui=True):
        """Test multiple custom objects with different scales"""
        
        all_results = {}
        overall_tests = 0
        overall_successes = 0
        
        for config in object_configs:
            obj_name = config['name']
            scale = config.get('scale', 1.0)
            num_tests = config.get('num_tests', 5)
            
            result = self.test_single_custom_object(
                obj_name=obj_name,
                scale=scale,
                num_tests=num_tests,
                gui=gui
            )
            
            all_results[obj_name] = result
            overall_tests += result['total_tests']
            overall_successes += result['success_count']
        
        # Print overall summary
        overall_rate = overall_successes / overall_tests * 100 if overall_tests > 0 else 0
        
        print(f"\n{'='*80}")
        print("COMPREHENSIVE CUSTOM OBJECT TEST RESULTS")
        print(f"{'='*80}")
        
        print(f"{'Object':<25} {'Scale':<8} {'Success Rate':<15} {'Tests':<10}")
        print("-" * 70)
        
        for obj_name, result in all_results.items():
            scale = result['scale']
            success_rate = result['success_rate']
            success_count = result['success_count']
            total_tests = result['total_tests']
            
            print(f"{obj_name:<25} {scale:<8} {success_rate:>6.1f}% {success_count:>3}/{total_tests:<6}")
        
        print("-" * 70)
        print(f"{'OVERALL':<25} {'Mixed':<8} {overall_rate:>6.1f}% {overall_successes:>3}/{overall_tests:<6}")
        
        return all_results
    
    def test_scale_sensitivity(self, obj_name, scales=[0.5, 1.0, 1.5, 2.0], tests_per_scale=3):
        """Test how object scale affects grasping performance"""
        
        print(f"\n{'='*60}")
        print(f"SCALE SENSITIVITY TEST: {obj_name.upper()}")
        print(f"{'='*60}")
        
        scale_results = {}
        
        for scale in scales:
            print(f"\nTesting scale: {scale}x")
            
            result = self.test_single_custom_object(
                obj_name=obj_name,
                scale=scale,
                num_tests=tests_per_scale,
                gui=True
            )
            
            scale_results[scale] = result
        
        # Print scale comparison
        print(f"\n{'='*60}")
        print(f"SCALE SENSITIVITY RESULTS FOR {obj_name.upper()}")
        print(f"{'='*60}")
        
        print(f"{'Scale':<10} {'Success Rate':<15} {'Tests':<10}")
        print("-" * 40)
        
        for scale, result in scale_results.items():
            success_rate = result['success_rate']
            success_count = result['success_count']
            total_tests = result['total_tests']
            
            print(f"{scale:<10} {success_rate:>6.1f}% {success_count:>3}/{total_tests:<6}")
        
        # Find optimal scale
        best_scale = max(scale_results.keys(), key=lambda s: scale_results[s]['success_rate'])
        best_rate = scale_results[best_scale]['success_rate']
        
        print(f"\nOptimal Scale: {best_scale}x (Success Rate: {best_rate:.1f}%)")
        
        return scale_results

def main():
    """Example usage of custom object testing"""
    
    # Create tester
    tester = CustomObjectTester()
    
    # Example 1: Test single object with different scales
    print("Example 1: Testing single object with multiple scales")
    
    # Replace 'your_object' with actual .obj filename (without .obj extension)
    scale_results = tester.test_scale_sensitivity(
        obj_name='your_object',  # Change this to your .obj filename
        scales=[0.5, 0.8, 1.0, 1.2, 1.5],
        tests_per_scale=3
    )
    
    # Example 2: Test multiple objects
    print("\n" + "="*80)
    print("Example 2: Testing multiple custom objects")
    
    object_configs = [
        {'name': 'object1', 'scale': 1.0, 'num_tests': 5},
        {'name': 'object2', 'scale': 0.8, 'num_tests': 5},
        {'name': 'object3', 'scale': 1.2, 'num_tests': 5},
    ]
    
    # Uncomment and modify object names to test your objects
    # multi_results = tester.test_multiple_custom_objects(object_configs, gui=True)
    
    print("\nTesting complete!")

if __name__ == "__main__":
    main()
