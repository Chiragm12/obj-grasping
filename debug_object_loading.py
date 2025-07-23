from src.pybullet_simulation import PyBulletGraspSimulation
import time

def debug_object_loading():
    print("=== DEBUG: Object Loading Test ===")
    
    # Create simulation with GUI
    sim = PyBulletGraspSimulation(gui=True)
    
    # List available custom objects
    available_objects = sim.list_custom_objects()
    print(f"Available objects: {available_objects}")
    
    if 'bowl' not in available_objects:
        print("❌ 'bowl' not found in available objects")
        print("Make sure bowl.obj is in the custom_objects/ directory")
        sim.close()
        return
    
    # Try to load the bowl
    print("\nAttempting to load bowl...")
    obj_id, obj_name = sim.load_custom_object('bowl', scale=1.0)
    
    if obj_id is not None:
        print(f"✅ Successfully loaded: {obj_name} (ID: {obj_id})")
        
        # Keep the simulation running to view the object
        print("Object should be visible in PyBullet GUI. Waiting 10 seconds...")
        time.sleep(10)
    else:
        print("❌ Failed to load bowl object")
    
    sim.close()

if __name__ == "__main__":
    debug_object_loading()
