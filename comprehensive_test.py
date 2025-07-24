import os
import time
import numpy as np
import pickle
import matplotlib.pyplot as plt
from datetime import datetime
from src.grasp_execution import GraspExecutor
from src.pybullet_simulation import PyBulletGraspSimulation
import pybullet as p

class ComprehensiveObjectTester:
    def __init__(self, model_path='models/grasp_model_final.pth'):
        self.executor = GraspExecutor(model_path)
        self.results = {}
        self.test_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(f"[SYSTEM] Comprehensive Object Testing Suite")
        print(f"[INFO] Test Session: {self.test_timestamp}")
        print(f"[INFO] Model: {model_path}")
        print("="*60)
        
    def test_all_custom_objects(self, tests_per_object=10, gui=False):
        """Test all objects in custom_objects folder"""
        
        # Initialize simulation
        sim = PyBulletGraspSimulation(gui=gui)
        
        # Get all available custom objects
        available_objects = sim.list_custom_objects()
        
        if not available_objects:
            print("[ERROR] No custom objects found in custom_objects/ folder!")
            sim.close()
            return {}
        
        print(f"[INFO] Found {len(available_objects)} custom objects:")
        for i, obj in enumerate(available_objects, 1):
            print(f"   {i}. {obj}")
        print()
        
        # Test each object
        all_results = {}
        overall_tests = 0
        overall_successes = 0
        
        for obj_idx, obj_name in enumerate(available_objects, 1):
            print(f"[TEST] Testing Object {obj_idx}/{len(available_objects)}: {obj_name.upper()}")
            print("-" * 50)
            
            # Find optimal scale for this object
            optimal_scale = self.find_optimal_scale(sim, obj_name)
            
            if optimal_scale is None:
                print(f"[SKIP] Could not load {obj_name}, skipping...")
                continue
            
            # Test this object multiple times
            object_results = self.test_single_object(
                sim, obj_name, optimal_scale, tests_per_object, gui
            )
            
            all_results[obj_name] = object_results
            overall_tests += object_results['total_tests']
            overall_successes += object_results['success_count']
            
            # Print object summary
            success_rate = object_results['success_rate']
            print(f"[RESULT] {obj_name}: {success_rate:.1f}% success rate")
            print()
        
        sim.close()
        
        # Calculate overall statistics
        overall_rate = overall_successes / overall_tests * 100 if overall_tests > 0 else 0
        
        # Store comprehensive results
        self.results = {
            'timestamp': self.test_timestamp,
            'overall_stats': {
                'total_objects': len(all_results),  # Only count successfully tested objects
                'total_tests': overall_tests,
                'total_successes': overall_successes,
                'overall_success_rate': overall_rate,
                'tests_per_object': tests_per_object
            },
            'object_results': all_results,
            'object_list': list(all_results.keys())  # Only successfully tested objects
        }
        
        # Generate comprehensive report
        self.generate_report()
        self.create_visualizations()
        
        return self.results
    
    def find_optimal_scale(self, sim, obj_name):
        """Find the optimal scale for an object with flexible criteria"""
        print(f"   [SCALE] Finding optimal scale for {obj_name}...")
        
        scales_to_try = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        successful_scales = []
        
        for scale in scales_to_try:
            try:
                obj_id, loaded_name = sim.load_custom_object(obj_name, scale=scale)
                
                if obj_id is not None:
                    # Check object size
                    aabb_min, aabb_max = p.getAABB(obj_id)
                    size = np.array(aabb_max) - np.array(aabb_min)
                    avg_size = np.mean(size)
                    
                    # More flexible size criteria - allow objects up to 50cm
                    if 0.02 <= avg_size <= 0.5:  # Expanded from 0.3 to 0.5
                        successful_scales.append((scale, avg_size))
                        print(f"   [OK] Scale {scale}: ACCEPTABLE (size: {avg_size:.3f}m)")
                    else:
                        print(f"   [REJECT] Scale {scale}: size {avg_size:.3f}m outside range [0.02-0.50]m")
                else:
                    print(f"   [FAIL] Scale {scale}: FAILED to load")
                
            except Exception as e:
                print(f"   [ERROR] Scale {scale}: {str(e)}")
                continue
        
        if successful_scales:
            # Choose scale closest to 15cm average size, but allow reasonable larger objects
            best_scale = min(successful_scales, 
                           key=lambda x: abs(x[1] - 0.15) if x[1] <= 0.3 else abs(x[1] - 0.25))
            print(f"   [OPTIMAL] Scale: {best_scale[0]} (size: {best_scale[1]:.3f}m)")
            return best_scale[0]
        else:
            print(f"   [NONE] No suitable scale found for {obj_name}")
            return None
    
    def test_single_object(self, sim, obj_name, scale, num_tests, gui):
        """Test a single object multiple times"""
        results = {
            'object_name': obj_name,
            'scale_used': scale,
            'tests': [],
            'success_count': 0,
            'total_tests': 0,
            'success_rate': 0.0,
            'grasp_positions': [],
            'grasp_orientations': [],
            'successful_grasps': [],
            'failed_grasps': []
        }
        
        for test_num in range(num_tests):
            if gui:
                print(f"   [TEST] Test {test_num + 1}/{num_tests}")
            
            # Load object
            obj_id, loaded_name = sim.load_custom_object(obj_name, scale=scale)
            
            if obj_id is None:
                print(f"   [ERROR] Failed to load {obj_name} for test {test_num + 1}")
                continue
            
            # Capture depth image
            depth_image, rgb_image = sim.capture_depth_image()
            
            # Predict grasp
            predicted_grasp = self.executor.predict_grasp(depth_image)
            
            # Execute grasp
            success = sim.execute_grasp(predicted_grasp, obj_id)
            
            # Store result
            test_result = {
                'test_id': test_num + 1,
                'predicted_grasp': predicted_grasp.copy(),
                'success': success,
                'position': predicted_grasp[:3].copy(),
                'orientation': predicted_grasp[3:].copy()
            }
            
            results['tests'].append(test_result)
            results['total_tests'] += 1
            
            if success:
                results['success_count'] += 1
                results['successful_grasps'].append(predicted_grasp.copy())
                if gui:
                    print(f"   [SUCCESS] Test {test_num + 1}: SUCCESS")
            else:
                results['failed_grasps'].append(predicted_grasp.copy())
                if gui:
                    print(f"   [FAIL] Test {test_num + 1}: FAILED")
            
            # Store position and orientation data
            results['grasp_positions'].append(predicted_grasp[:3])
            results['grasp_orientations'].append(predicted_grasp[3:])
            
            if gui:
                time.sleep(1)  # Pause for visualization
        
        # Calculate statistics
        if results['total_tests'] > 0:
            results['success_rate'] = results['success_count'] / results['total_tests'] * 100
        
        return results
    
    def generate_report(self):
        """Generate comprehensive text report with proper encoding"""
        report_filename = f"test_report_{self.test_timestamp}.txt"
        
        # Use UTF-8 encoding to handle all characters properly
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("COMPREHENSIVE ROBOTIC GRASPING TEST REPORT\n")
            f.write("="*80 + "\n")
            f.write(f"Test Session: {self.test_timestamp}\n")
            f.write(f"Model: models/grasp_model_final.pth\n")
            f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Overall Statistics
            overall = self.results['overall_stats']
            f.write("OVERALL PERFORMANCE SUMMARY\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total Objects Tested: {overall['total_objects']}\n")
            f.write(f"Total Tests Performed: {overall['total_tests']}\n")
            f.write(f"Total Successful Grasps: {overall['total_successes']}\n")
            f.write(f"Overall Success Rate: {overall['overall_success_rate']:.1f}%\n")
            f.write(f"Tests per Object: {overall['tests_per_object']}\n\n")
            
            # Detailed Object Results
            f.write("DETAILED OBJECT PERFORMANCE\n")
            f.write("-" * 40 + "\n")
            f.write(f"{'Object':<15} {'Success Rate':<12} {'Tests':<8} {'Scale':<8}\n")
            f.write("-" * 50 + "\n")
            
            for obj_name, obj_results in self.results['object_results'].items():
                success_rate = obj_results['success_rate']
                success_count = obj_results['success_count']
                total_tests = obj_results['total_tests']
                scale = obj_results['scale_used']
                
                f.write(f"{obj_name:<15} {success_rate:>6.1f}% {success_count:>3}/{total_tests:<4} {scale:<8}\n")
            
            f.write("\n")
            
            # Performance Analysis
            f.write("PERFORMANCE ANALYSIS\n")
            f.write("-" * 40 + "\n")
            
            success_rates = []
            for obj_results in self.results['object_results'].values():
                success_rates.append(obj_results['success_rate'])
            
            if success_rates:
                best_obj = max(self.results['object_results'].items(), key=lambda x: x[1]['success_rate'])
                worst_obj = min(self.results['object_results'].items(), key=lambda x: x[1]['success_rate'])
                
                f.write(f"Best Performing Object: {best_obj[0]} ({best_obj[1]['success_rate']:.1f}%)\n")
                f.write(f"Worst Performing Object: {worst_obj[0]} ({worst_obj[1]['success_rate']:.1f}%)\n")
                f.write(f"Average Success Rate: {np.mean(success_rates):.1f}%\n")
                f.write(f"Standard Deviation: {np.std(success_rates):.1f}%\n\n")
            
            # Conclusion - Use ASCII characters instead of Unicode emojis
            f.write("CONCLUSION\n")
            f.write("-" * 40 + "\n")
            if overall['overall_success_rate'] >= 70:
                f.write("[EXCELLENT] System demonstrates high-quality grasping performance\n")
            elif overall['overall_success_rate'] >= 50:
                f.write("[GOOD] System shows solid grasping capabilities\n")
            elif overall['overall_success_rate'] >= 30:
                f.write("[MODERATE] System has basic grasping functionality\n")
            else:
                f.write("[NEEDS IMPROVEMENT] System requires optimization\n")
            
            f.write(f"The robotic grasping system achieved an overall success rate of {overall['overall_success_rate']:.1f}% ")
            f.write(f"across {overall['total_objects']} different objects with {overall['total_tests']} total test attempts.\n")
        
        print(f"[SAVE] Comprehensive report saved: {report_filename}")
        return report_filename
    
    def create_visualizations(self):
        """Create visualization plots for the report"""
        
        # Extract data for plotting
        object_names = []
        success_rates = []
        test_counts = []
        
        for obj_name, obj_results in self.results['object_results'].items():
            object_names.append(obj_name)
            success_rates.append(obj_results['success_rate'])
            test_counts.append(obj_results['total_tests'])
        
        if not object_names:  # No data to plot
            print("[WARNING] No data available for visualization")
            return None
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Robotic Grasping Test Results - {self.test_timestamp}', fontsize=16, fontweight='bold')
        
        # 1. Success Rate by Object (Bar Chart)
        bars = ax1.bar(range(len(object_names)), success_rates, 
                      color=['green' if sr >= 70 else 'orange' if sr >= 50 else 'red' for sr in success_rates])
        ax1.set_xlabel('Objects')
        ax1.set_ylabel('Success Rate (%)')
        ax1.set_title('Success Rate by Object')
        ax1.set_xticks(range(len(object_names)))
        ax1.set_xticklabels(object_names, rotation=45, ha='right')
        ax1.set_ylim(0, 105)
        
        # Add value labels on bars
        for bar, rate in zip(bars, success_rates):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 2. Success Distribution (Histogram)
        ax2.hist(success_rates, bins=max(1, len(success_rates)//2), alpha=0.7, color='skyblue', edgecolor='black')
        ax2.set_xlabel('Success Rate (%)')
        ax2.set_ylabel('Number of Objects')
        ax2.set_title('Distribution of Success Rates')
        ax2.axvline(np.mean(success_rates), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(success_rates):.1f}%')
        ax2.legend()
        
        # 3. Overall Statistics (Text Summary)
        ax3.axis('off')
        overall_stats = self.results['overall_stats']
        stats_text = f"""
OVERALL STATISTICS

Total Objects: {overall_stats['total_objects']}
Total Tests: {overall_stats['total_tests']}
Success Rate: {overall_stats['overall_success_rate']:.1f}%

PERFORMANCE BREAKDOWN
Excellent (>=70%): {sum(1 for sr in success_rates if sr >= 70)} objects
Good (50-69%): {sum(1 for sr in success_rates if 50 <= sr < 70)} objects
Poor (<50%): {sum(1 for sr in success_rates if sr < 50)} objects

Best Object: {object_names[np.argmax(success_rates)]} ({max(success_rates):.1f}%)
Worst Object: {object_names[np.argmin(success_rates)]} ({min(success_rates):.1f}%)
        """
        ax3.text(0.1, 0.9, stats_text, transform=ax3.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        # 4. Success Rate Trend
        ax4.plot(range(len(object_names)), success_rates, 'o-', linewidth=2, markersize=8)
        ax4.set_xlabel('Object Index')
        ax4.set_ylabel('Success Rate (%)')
        ax4.set_title('Success Rate Across Objects')
        ax4.set_xticks(range(len(object_names)))
        ax4.set_xticklabels([f"{i+1}" for i in range(len(object_names))])
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=np.mean(success_rates), color='r', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        # Save the plot
        plot_filename = f"test_results_{self.test_timestamp}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[SAVE] Visualization saved: {plot_filename}")
        return plot_filename
    
    def save_detailed_results(self):
        """Save detailed results to pickle file for further analysis"""
        results_filename = f"detailed_results_{self.test_timestamp}.pkl"
        
        with open(results_filename, 'wb') as f:
            pickle.dump(self.results, f)
        
        print(f"[SAVE] Detailed results saved: {results_filename}")
        return results_filename

def main():
    """Main function to run comprehensive testing"""
    print("[START] Starting Comprehensive Multi-Object Grasp Testing")
    print("This will test all objects in your custom_objects/ folder")
    print()
    
    # Configuration
    TESTS_PER_OBJECT = 10  # Adjust this number based on your time constraints
    SHOW_GUI = False       # Set to True if you want to see each test visually
    
    # Check if custom objects exist
    if not os.path.exists("custom_objects") or len(os.listdir("custom_objects")) == 0:
        print("[ERROR] No objects found in custom_objects/ folder!")
        print("Please add your .obj files to the custom_objects/ directory")
        return
    
    # Create tester and run comprehensive tests
    tester = ComprehensiveObjectTester()
    
    print(f"[CONFIG] Configuration:")
    print(f"   Tests per object: {TESTS_PER_OBJECT}")
    print(f"   GUI visualization: {'ON' if SHOW_GUI else 'OFF'}")
    print(f"   Expected duration: {len(os.listdir('custom_objects')) * 2} minutes")
    print()
    
    input("Press Enter to start testing...")
    
    # Run the comprehensive test
    start_time = time.time()
    results = tester.test_all_custom_objects(
        tests_per_object=TESTS_PER_OBJECT,
        gui=SHOW_GUI
    )
    end_time = time.time()
    
    # Final summary
    print("\n" + "="*80)
    print("[COMPLETE] COMPREHENSIVE TESTING COMPLETED!")
    print("="*80)
    
    if results:
        overall = results['overall_stats']
        print(f"[RESULTS] FINAL RESULTS:")
        print(f"   Overall Success Rate: {overall['overall_success_rate']:.1f}%")
        print(f"   Objects Tested: {overall['total_objects']}")
        print(f"   Total Tests: {overall['total_tests']}")
        print(f"   Duration: {(end_time - start_time)/60:.1f} minutes")
        
        # Save detailed results
        tester.save_detailed_results()
        
        print(f"\n[FILES] Generated Files:")
        print(f"   Text Report: test_report_{tester.test_timestamp}.txt")
        print(f"   Visualizations: test_results_{tester.test_timestamp}.png")
        print(f"   Raw Data: detailed_results_{tester.test_timestamp}.pkl")
        
        print(f"\n[SUCCESS] All files ready for your project report!")
    
    else:
        print("[ERROR] No results generated - check your setup and try again.")

if __name__ == "__main__":
    main()
