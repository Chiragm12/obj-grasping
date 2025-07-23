from src.pybullet_simulation import PyBulletGraspSimulation
import time

def test_custom_object_loading():
    print("=== Testing Custom Object Loading ===")
    
    # Create simulation with GUI
    sim = PyBulletGraspSimulation(gui=True)
    
    # List available objects
    available_objects = sim.list_custom_objects()
    print(f"Available custom objects: {available_objects}")
    
    if not available_objects:
        print("❌ No custom objects found!")
        print("Please place your .obj files in the custom_objects/ directory")
        sim.close()
        return
    
    # Test loading the first available object
    test_object = available_objects[0]
    print(f"\nTesting object: {test_object}")
    
    # Try to load the object
    obj_id, obj_name = sim.load_custom_object(test_object, scale=1.0)
    
    if obj_id is not None:
        print(f"✅ Successfully loaded: {obj_name}")
        
        # Get object information
        info = sim.get_custom_object_info(test_object)
        if info:
            print(f"Object info:")
            print(f"  Position: {info['position']}")
            print(f"  Size: {info['size']}")
            print(f"  Scale used: {info['scale']}")
        
        # Keep simulation running to view the object
        print("\nObject should be visible in PyBullet GUI.")
        print("Press Ctrl+C to exit or wait 15 seconds...")
        
        try:
            time.sleep(15)
        except KeyboardInterrupt:
            print("\nExiting...")
    
    else:
        print("❌ Failed to load object")
        
        # Try the scale testing feature
        print("\nTrying automatic scale detection...")
        best_scale = sim.test_multiple_scales(test_object)
        
        if best_scale:
            print(f"Try using scale={best_scale} for this object")
    
    sim.close()

if __name__ == "__main__":
    test_custom_object_loading()
