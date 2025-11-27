from modular_drl_env.world.world import World
import numpy as np
import pybullet as pyb
from modular_drl_env.world.obstacles.shapes import Box, Sphere, Cylinder
from modular_drl_env.world.obstacles.ground_plate import GroundPlate
from modular_drl_env.world.obstacles.urdf_object import URDFObject
import pybullet_data as pyb_d
from modular_drl_env.util.pybullet_util import pybullet_util as pyb_u

__all__ = [
    'ManipulateObjects'
]

class ManipulateObjects(World):
    """
    A simple demonstration world with static obstacles.
    Great for testing and learning the framework.
    """

    def __init__(self, workspace_boundaries: list, 
                       sim_step: float,
                       sim_steps_per_env_step: int,
                       env_id: int,
                       assets_path: str,
                       num_boxes: int = 2,
                       num_spheres: int = 2,
                       add_table: bool = True,
                       table_dim:  list = [0.625,0.5,1.0]):  
        super().__init__(workspace_boundaries, sim_step, sim_steps_per_env_step, env_id, assets_path)
        
        self.num_boxes = num_boxes
        self.num_spheres = num_spheres
        self.add_table = add_table
        self.height, self.width, self.length  = table_dim  
        
        # Storage position for unused objects
        self.position_nowhere = np.array([0, 0, -10])
        
    def set_up(self):
        """
        Build all the world components.
        This is called once when the environment is created.
        """
        # 1. Add ground plane
        ground = GroundPlate()
        ground.build()
        
        # 2. Optional: Add a table
        if self.add_table:
            table = URDFObject(
                position=[0, 0.5, 0], 
                rotation=[0, 0, 0, 1],  # Use 'rotation' not 'orientation'
                trajectory=[], 
                sim_step=self.sim_step, 
                sim_steps_per_env_step=self.sim_steps_per_env_step, 
                velocity=0, 
                urdf_path=pyb_d.getDataPath() + "/table/table.urdf", 
                scale=1.0,
                seen_by_obstacle_sensor=False  # Make table invisible to collision detection
            )
            table.build()
            self.obstacle_objects.append(table)
            
        # 3. Pre-generate boxes at different positions
        for i in range(self.num_boxes):
            # Random size for each box
            half_extents = np.random.uniform(low=0.03, high=0.08, size=(3,)).tolist()
            
            # Create box (starts at position_nowhere)
            box = Box(
                position=self.position_nowhere.copy(),
                rotation=[0, 0, 0, 1],  # Note: it's 'rotation' not 'orientation'
                trajectory=[],  # Static obstacle (no movement)
                sim_step=self.sim_step,
                sim_steps_per_env_step=self.sim_steps_per_env_step,
                velocity=0,
                halfExtents=half_extents
            )
            box.build()
            self.obstacle_objects.append(box)
        
        # 4. Pre-generate spheres
        for i in range(self.num_spheres):
            # Random radius for each sphere
            radius = np.random.uniform(low=0.02, high=0.06)
            
            sphere = Sphere(
                position=self.position_nowhere.copy(),
                trajectory=[],  # Sphere doesn't take orientation
                sim_step=self.sim_step,
                sim_steps_per_env_step=self.sim_steps_per_env_step,
                velocity=0,
                radius=radius
            )
            sphere.build()
            self.obstacle_objects.append(sphere)
        
        print(f"✅ ManipulateObjects world created with {self.num_boxes} boxes and {self.num_spheres} spheres on table")

    def reset(self, success_rate: float):
        """
        Reset the world for a new episode.
        This places obstacles at new random positions within the workspace.
        """
        # Clear previous active objects
        for obj in self.active_objects:
            obj.move_base(self.position_nowhere)
        self.active_objects = []
        # Reset goal targets
        self.position_targets = []
        self.rotation_targets = []
        self.joints_targets = []
        
        # Randomly select and place obstacles
        available_obstacles = [obj for obj in self.obstacle_objects if not isinstance(obj, URDFObject)]
        num_active = min(len(available_obstacles), self.num_boxes + self.num_spheres)
        selected_obstacles = np.random.choice(available_obstacles, size=num_active, replace=False)
        
        for obstacle in selected_obstacles:
            # Generate random position within workspace boundaries
            x = np.random.uniform(0.0 + 0.1, self.width - 0.1)
            y = np.random.uniform(0.0 + 0.1, self.length - 0.1)
            #z = np.random.uniform(self.height + 0.1, self.z_max - 0.2)
            z = self.height + 0.05
            # Random orientation (only matters for boxes, not spheres)
            orientation = pyb.getQuaternionFromEuler([
                0,  # No roll - keep objects upright
                0,  # No pitch - keep objects upright  
                np.random.uniform(-np.pi, np.pi)  # Random yaw rotation
            ])
            
            # Move obstacle to position with orientation
            obstacle.move_base(
                new_base_position=np.array([x, y, z]),
                new_base_rotation=np.array(orientation)
            )
            
            self.active_objects.append(obstacle)
        
        # Select one random object as the target to reach/grasp
        if len(self.active_objects) > 0:
            self.target_object = np.random.choice(self.active_objects)
            target_pos = np.array(self.target_object.position)
            print(f"🎯 Target object at {target_pos}")
        else:
            # Fallback: random position if no objects
            target_pos = np.array([
                np.random.uniform(0.0 + 0.15, self.width - 0.15),
                np.random.uniform(0.0 + 0.15, self.length - 0.15),
                self.height
            ])
            self.target_object = None
        
        self.position_targets.append(target_pos)
        self.rotation_targets.append([0, 0, 0, 1])  # Default orientation
        
        print(f"🔄 Episode reset: {len(self.active_objects)} active obstacles, target at {target_pos}")
    
    def update(self):
        """
        Update the world state each step.
        For this simple world with static obstacles, we don't need to do anything.
        If you had moving obstacles, you would update their positions here.
        """
        # No dynamic obstacles in this simple world, so nothing to update
        pass
        
    def build_visual_aux(self):
        """
        Optional: Add visual aids for debugging (like showing the goal position).
        """
        # First, draw the workspace boundaries (parent class method)
        super().build_visual_aux()
        
        # Draw a small sphere at the goal position
        if len(self.position_targets) > 0:
            goal_pos = self.position_targets[0]
            visual_shape = pyb.createVisualShape(
                shapeType=pyb.GEOM_SPHERE,
                radius=0.03,
                rgbaColor=[0, 1, 0, 0.5]  # Green semi-transparent
            )
            goal_marker = pyb.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=visual_shape,
                basePosition=goal_pos
            )
            self.aux_objects.append(goal_marker)
