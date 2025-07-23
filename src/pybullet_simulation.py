import pybullet as p
import pybullet_data
import numpy as np
import time
import os
from .utils import load_config, process_depth_image, euler_to_quaternion
from .custom_object_loader import CustomObjectLoader

class PyBulletGraspSimulation:
    def __init__(self, gui=False, config_path="config/config.yaml"):
        self.config = load_config(config_path)
        self.data_config = self.config['data_generation']
        self.gripper_config = self.config['gripper']
        self.custom_loader = CustomObjectLoader()
        # Initialize PyBullet
        if gui:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # Load plane
        self.plane_id = p.loadURDF("plane.urdf")
        
        # Camera parameters
        self.camera_distance = self.data_config['camera_distance']
        self.image_size = tuple(self.data_config['image_size'])
        self.fov = self.data_config['fov']
        
        # Calculate camera intrinsics
        self.setup_camera()
        
    def setup_camera(self):
        """Setup camera parameters"""
        width, height = self.image_size
        aspect = width / height
        near = 0.1
        far = 2.0
        
        self.projection_matrix = p.computeProjectionMatrixFOV(
            fov=self.fov,
            aspect=aspect,
            nearVal=near,
            farVal=far
        )
        
        # Camera intrinsics for depth processing
        fov_rad = np.deg2rad(self.fov)
        self.fx = width / (2 * np.tan(fov_rad / 2))
        self.fy = height / (2 * np.tan(fov_rad / 2))
        self.cx = width / 2
        self.cy = height / 2
    
    def create_basic_objects(self):
        """Create basic geometric objects programmatically"""
        objects = {}
        
        # Create Box/Cube
        box_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.05])
        box_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.05], 
                                       rgbaColor=[1, 0, 0, 1])
        objects['cube'] = (box_collision, box_visual)
        
        # Create Sphere
        sphere_collision = p.createCollisionShape(p.GEOM_SPHERE, radius=0.05)
        sphere_visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.05, 
                                          rgbaColor=[0, 1, 0, 1])
        objects['sphere'] = (sphere_collision, sphere_visual)
        
        # Create Cylinder
        cylinder_collision = p.createCollisionShape(p.GEOM_CYLINDER, 
                                                  radius=0.04, height=0.08)
        cylinder_visual = p.createVisualShape(p.GEOM_CYLINDER, 
                                            radius=0.04, length=0.08,
                                            rgbaColor=[0, 0, 1, 1])
        objects['cylinder'] = (cylinder_collision, cylinder_visual)
        
        # Create Capsule
        capsule_collision = p.createCollisionShape(p.GEOM_CAPSULE, 
                                                 radius=0.03, height=0.06)
        capsule_visual = p.createVisualShape(p.GEOM_CAPSULE, 
                                           radius=0.03, length=0.06,
                                           rgbaColor=[1, 1, 0, 1])
        objects['capsule'] = (capsule_collision, capsule_visual)
        
        return objects
    
    def load_random_object(self):
        """Load a random object from programmatically created objects"""
        # Remove any existing objects (except plane)
        num_bodies = p.getNumBodies()
        bodies_to_remove = []
        
        for i in range(num_bodies):
            body_id = p.getBodyUniqueId(i)
            if body_id != self.plane_id:
                bodies_to_remove.append(body_id)
        
        # Remove collected bodies
        for body_id in bodies_to_remove:
            p.removeBody(body_id)
        
        # Create object definitions
        objects = self.create_basic_objects()
        
        # Try to load available URDF files as backup
        urdf_objects = []
        data_path = pybullet_data.getDataPath()
        
        # Check for common URDF files
        possible_urdfs = [
            "duck_vhacd.urdf",
            "teddy_vhacd.urdf", 
            "r2d2.urdf",
            "tray/traybox.urdf"
        ]
        
        for urdf in possible_urdfs:
            urdf_path = os.path.join(data_path, urdf)
            if os.path.exists(urdf_path):
                urdf_objects.append(urdf)
        
        # Combine programmatic objects with available URDFs
        all_objects = list(objects.keys()) + urdf_objects
        
        # Select random object
        object_name = np.random.choice(all_objects)
        
        # Random position and orientation
        pos_x = np.random.uniform(-0.2, 0.2)
        pos_y = np.random.uniform(-0.2, 0.2)
        pos_z = np.random.uniform(0.1, 0.2)  # Higher to avoid collision with plane
        
        roll = np.random.uniform(-np.pi, np.pi)
        pitch = np.random.uniform(-np.pi, np.pi)
        yaw = np.random.uniform(-np.pi, np.pi)
        
        quat = euler_to_quaternion(roll, pitch, yaw)
        
        try:
            if object_name in objects:
                # Create programmatic object
                collision_shape, visual_shape = objects[object_name]
                
                object_id = p.createMultiBody(
                    baseMass=1.0,
                    baseCollisionShapeIndex=collision_shape,
                    baseVisualShapeIndex=visual_shape,
                    basePosition=[pos_x, pos_y, pos_z],
                    baseOrientation=quat
                )
                
                print(f"Created programmatic object: {object_name}")
                
            else:
                # Load URDF object
                object_id = p.loadURDF(
                    object_name,
                    basePosition=[pos_x, pos_y, pos_z],
                    baseOrientation=quat
                )
                
                print(f"Loaded URDF object: {object_name}")
            
            # Let object settle
            for _ in range(50):
                p.stepSimulation()
            
            return object_id, object_name
            
        except Exception as e:
            print(f"Failed to create/load {object_name}: {e}")
            
            # Fallback to simple cube
            try:
                box_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.05])
                box_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.05], 
                                               rgbaColor=[0.5, 0.5, 0.5, 1])
                
                object_id = p.createMultiBody(
                    baseMass=1.0,
                    baseCollisionShapeIndex=box_collision,
                    baseVisualShapeIndex=box_visual,
                    basePosition=[0, 0, 0.1],
                    baseOrientation=[0, 0, 0, 1]
                )
                
                print("Created fallback cube object")
                return object_id, "fallback_cube"
                
            except Exception as fallback_error:
                print(f"Even fallback failed: {fallback_error}")
                return None, None
    
    def capture_depth_image(self, camera_pos=None):
        """Capture depth image from camera"""
        if camera_pos is None:
            # Default camera position
            camera_pos = [0, 0, self.camera_distance]
        
        target_pos = [0, 0, 0]
        up_vector = [0, 1, 0]
        
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=camera_pos,
            cameraTargetPosition=target_pos,
            cameraUpVector=up_vector
        )
        
        # Capture image
        width, height, rgb_img, depth_img, seg_img = p.getCameraImage(
            width=self.image_size[0],
            height=self.image_size[1],
            viewMatrix=view_matrix,
            projectionMatrix=self.projection_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        
        # Process depth image
        depth_buffer = np.array(depth_img, dtype=np.float32).reshape(height, width)
        depth_image = process_depth_image(depth_buffer, near=0.1, far=2.0)
        
        return depth_image, rgb_img
    
    def generate_grasp_candidates(self, object_id, num_candidates=20):
        """Generate grasp candidates for object"""
        # Get object position and AABB
        pos, orn = p.getBasePositionAndOrientation(object_id)
        aabb_min, aabb_max = p.getAABB(object_id)
        
        grasp_candidates = []
        
        for _ in range(num_candidates):
            # Random position around object
            grasp_x = np.random.uniform(aabb_min[0], aabb_max[0])
            grasp_y = np.random.uniform(aabb_min[1], aabb_max[1])
            grasp_z = np.random.uniform(aabb_min[2] + 0.02, aabb_max[2] + 0.1)
            
            # Random orientation (focusing on top-down grasps)
            roll = np.random.uniform(-np.pi/4, np.pi/4)
            pitch = np.random.uniform(-np.pi/4, np.pi/4)  
            yaw = np.random.uniform(-np.pi, np.pi)
            
            grasp_pose = np.array([grasp_x, grasp_y, grasp_z, roll, pitch, yaw])
            grasp_candidates.append(grasp_pose)
        
        return grasp_candidates
    
    def evaluate_grasp_quality(self, grasp_pose, object_id):
        """Improved grasp quality evaluation"""
        pos, orn = p.getBasePositionAndOrientation(object_id)
        aabb_min, aabb_max = p.getAABB(object_id)
        
        obj_center = np.array(pos)
        grasp_pos = grasp_pose[:3]
        object_size = np.array(aabb_max) - np.array(aabb_min)
        
        # Distance to object center (normalized by object size)
        distance_to_center = np.linalg.norm(grasp_pos - obj_center)
        size_normalized_distance = distance_to_center / np.linalg.norm(object_size)
        
        # Height appropriateness
        if object_size[2] < 0.08:  # Small/flat objects
            # Prefer top-down grasps
            height_score = 1.0 if grasp_pos[2] > aabb_max[2] else 0.5
        else:  # Taller objects
            # Prefer mid-height grasps
            mid_height = (aabb_min[2] + aabb_max[2]) / 2
            height_diff = abs(grasp_pos[2] - mid_height) / object_size[2]
            height_score = 1.0 - height_diff
        
        # Approach angle penalty (prefer more vertical approaches for small objects)
        roll, pitch = grasp_pose[3], grasp_pose[4]
        angle_penalty = abs(roll) + abs(pitch)
        
        if object_size[2] < 0.08:  # Small objects
            angle_penalty *= 2.0  # Stronger penalty for tilted grasps
        
        # Combined quality score
        quality = (
            (1.0 / (1.0 + size_normalized_distance * 2)) * 0.4 +  # Distance component
            height_score * 0.4 +                                   # Height component
            max(0, 1.0 - angle_penalty / np.pi) * 0.2             # Angle component
        )
        
        return quality
    
    def select_best_grasp(self, grasp_candidates, object_id):
        """Select best grasp from candidates"""
        best_grasp = None
        best_quality = -1
        
        for grasp in grasp_candidates:
            quality = self.evaluate_grasp_quality(grasp, object_id)
            if quality > best_quality:
                best_quality = quality
                best_grasp = grasp
        
        return best_grasp, best_quality
    
    def execute_grasp(self, grasp_pose, object_id):
        """Execute grasp in simulation (for testing)"""
        # Create simple gripper
        gripper_pos = grasp_pose[:3]
        gripper_orn = euler_to_quaternion(*grasp_pose[3:6])
        
        # Move to pre-grasp position
        pre_grasp_pos = gripper_pos.copy()
        pre_grasp_pos[2] += self.gripper_config['approach_distance']
        
        # This is a simplified grasp execution
        # In practice, you'd need a proper gripper model
        success = self.check_grasp_feasibility(grasp_pose, object_id)
        
        return success
    
    def check_grasp_feasibility(self, grasp_pose, object_id):
        """Check if grasp is feasible (simplified)"""
        # Get object AABB
        aabb_min, aabb_max = p.getAABB(object_id)
        grasp_pos = grasp_pose[:3]
        
        # Check if grasp position is within reasonable bounds
        if (aabb_min[0] <= grasp_pos[0] <= aabb_max[0] and
            aabb_min[1] <= grasp_pos[1] <= aabb_max[1] and
            grasp_pos[2] >= aabb_min[2]):
            return True
        
        return False
    
    def clear_objects(self):
        """Remove all objects except the plane"""
        num_bodies = p.getNumBodies()
        bodies_to_remove = []
        
        for i in range(num_bodies):
            body_id = p.getBodyUniqueId(i)
            if body_id != self.plane_id:
                bodies_to_remove.append(body_id)
        
        for body_id in bodies_to_remove:
            p.removeBody(body_id)
    def load_custom_object(self, obj_name, scale=1.0):
        """Load a custom .obj file for testing"""
        
        # Clear existing objects first
        self.clear_objects()
        
        # Load the custom object
        object_id, loaded_name = self.custom_loader.load_obj_file(
            obj_name=obj_name,
            scale=scale,
            mass=1.0,
            position=[0, 0, 0.2]  # Higher position for better visibility
        )
        
        if object_id is not None:
            print(f"Loaded custom object: {loaded_name}")
            
            # Get object info for debugging
            pos, orn = p.getBasePositionAndOrientation(object_id)
            aabb_min, aabb_max = p.getAABB(object_id)
            
            print(f"Object position: {pos}")
            print(f"Object AABB: {aabb_min} to {aabb_max}")
            
            return object_id, loaded_name
        else:
            print(f"Failed to load custom object: {obj_name}")
            return None, None

    def list_custom_objects(self):
        """List available custom .obj files"""
        return self.custom_loader.list_available_objects()

    def load_specific_test_object(self, obj_name):
        """Load a specific test object (for compatibility with test scripts)"""
        return self.load_custom_object(obj_name, scale=1.0)

    def get_custom_object_info(self, obj_name):
        """Get information about a loaded custom object"""
        if obj_name in self.custom_loader.loaded_objects:
            obj_info = self.custom_loader.loaded_objects[obj_name]
            object_id = obj_info['id']
            
            # Get current position and orientation
            pos, orn = p.getBasePositionAndOrientation(object_id)
            
            # Get bounding box
            aabb_min, aabb_max = p.getAABB(object_id)
            
            return {
                'id': object_id,
                'position': pos,
                'orientation': orn,
                'aabb_min': aabb_min,
                'aabb_max': aabb_max,
                'size': np.array(aabb_max) - np.array(aabb_min),
                'path': obj_info['path'],
                'scale': obj_info['scale'],
                'mass': obj_info['mass']
            }
        
        return None

    def test_multiple_scales(self, obj_name, scales=[0.01, 0.1, 1.0, 10.0]):
        """Test loading an object with different scales to find the best one"""
        print(f"Testing different scales for {obj_name}:")
        
        successful_scales = []
        
        for scale in scales:
            print(f"  Trying scale {scale}...")
            
            try:
                obj_id, loaded_name = self.load_custom_object(obj_name, scale=scale)
                
                if obj_id is not None:
                    # Check if object is reasonably sized
                    aabb_min, aabb_max = p.getAABB(obj_id)
                    size = np.array(aabb_max) - np.array(aabb_min)
                    
                    # Object should be between 2cm and 50cm in each dimension
                    if all(0.02 <= s <= 0.5 for s in size):
                        successful_scales.append((scale, size))
                        print(f"    ✅ Scale {scale} successful - Size: {size}")
                    else:
                        print(f"    ⚠️ Scale {scale} loaded but size seems wrong: {size}")
                else:
                    print(f"    ❌ Scale {scale} failed to load")
                    
            except Exception as e:
                print(f"    ❌ Scale {scale} error: {e}")
        
        if successful_scales:
            # Return the scale that gives the most reasonable size (closest to 10cm)
            target_size = 0.1
            best_scale = min(successful_scales, 
                            key=lambda x: abs(np.mean(x[1]) - target_size))
            
            print(f"\nRecommended scale for {obj_name}: {best_scale[0]}")
            return best_scale[0]
        else:
            print(f"\n❌ No successful scales found for {obj_name}")
            return None



    def close(self):
        """Close simulation"""
        p.disconnect()

if __name__ == "__main__":
    # Test simulation
    sim = PyBulletGraspSimulation(gui=True)
    
    # Load object and capture image
    obj_id, obj_name = sim.load_random_object()
    if obj_id:
        depth_img, rgb_img = sim.capture_depth_image()
        
        # Generate and evaluate grasps
        candidates = sim.generate_grasp_candidates(obj_id)
        best_grasp, quality = sim.select_best_grasp(candidates, obj_id)
        
        print(f"Object: {obj_name}")
        print(f"Best grasp: {best_grasp}")
        print(f"Quality: {quality}")
        
        time.sleep(5)
    
    sim.close()
