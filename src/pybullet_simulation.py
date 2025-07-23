import pybullet as p
import pybullet_data
import numpy as np
import time
import os
from .utils import load_config, process_depth_image, euler_to_quaternion

class PyBulletGraspSimulation:
    def __init__(self, gui=False, config_path="config/config.yaml"):
        self.config = load_config(config_path)
        self.data_config = self.config['data_generation']
        self.gripper_config = self.config['gripper']
        
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
        """Simple grasp quality evaluation"""
        # Get object properties
        pos, orn = p.getBasePositionAndOrientation(object_id)
        aabb_min, aabb_max = p.getAABB(object_id)
        
        # Distance to object center
        obj_center = np.array(pos)
        grasp_pos = grasp_pose[:3]
        distance_to_center = np.linalg.norm(grasp_pos - obj_center)
        
        # Height above table
        height_score = grasp_pose[2] - aabb_min[2]
        
        # Simple scoring (higher is better)
        quality = 1.0 / (1.0 + distance_to_center) + height_score * 0.5
        
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
