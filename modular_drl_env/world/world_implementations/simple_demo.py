from modular_drl_env.world.world import World
import numpy as np
import pybullet as pyb
from modular_drl_env.world.obstacles.shapes import Box, Sphere, Cylinder
from modular_drl_env.world.obstacles.ground_plate import GroundPlate
from modular_drl_env.world.obstacles.urdf_object import URDFObject
import pybullet_data as pyb_d
from modular_drl_env.util.pybullet_util import pybullet_util as pyb_u

__all__ = [
    'SimpleDemoWorld'
]

class SimpleDemoWorld(World):
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
                       add_table: bool = True):
        super().__init__(workspace_boundaries, sim_step, sim_steps_per_env_step, env_id, assets_path)
        
        self.num_boxes = num_boxes
        self.num_spheres = num_spheres
        self.add_table = add_table
        
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
                orientation=[0, 0, 0, 1], 
                trajectory=[], 
                sim_step=self.sim_step, 
                sim_steps_per_env_step=self.sim_steps_per_env_step, 
                velocity=0, 
                urdf_path=pyb_d.getDataPath() + "/table/table.urdf", 
                scale=1.5
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
        
        print(f"✅ SimpleDemoWorld created with {self.num_boxes} boxes and {self.num_spheres} spheres")

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
            # Generate random position within workspace
            x = np.random.uniform(self.x_min + 0.1, self.x_max - 0.1)
            y = np.random.uniform(self.y_min + 0.1, self.y_max - 0.1)
            z = np.random.uniform(self.z_min + 0.1, self.z_max - 0.1)
            
            # Random orientation
            orientation = pyb.getQuaternionFromEuler([
                np.random.uniform(-np.pi, np.pi),
                np.random.uniform(-np.pi, np.pi),
                np.random.uniform(-np.pi, np.pi)
            ])
            
            # Move obstacle to position with orientation
            obstacle.move_base(
                new_base_position=np.array([x, y, z]),
                new_base_rotation=np.array(orientation)
            )
            
            self.active_objects.append(obstacle)
        
        # Generate a random goal position for the robot (away from obstacles)
        goal_found = False
        max_attempts = 100
        attempts = 0
        
        while not goal_found and attempts < max_attempts:
            goal_x = np.random.uniform(self.x_min + 0.15, self.x_max - 0.15)
            goal_y = np.random.uniform(self.y_min + 0.15, self.y_max - 0.15)
            goal_z = np.random.uniform(self.z_min + 0.15, self.z_max - 0.15)
            goal_pos = np.array([goal_x, goal_y, goal_z])
            
            # Check if goal is far enough from obstacles
            min_dist = float('inf')
            for obj in self.active_objects:
                dist = np.linalg.norm(goal_pos - np.array(obj.position))
                min_dist = min(min_dist, dist)
            
            if min_dist > 0.15:  # At least 15cm away from obstacles
                goal_found = True
            attempts += 1
        
        self.position_targets.append(goal_pos)
        self.rotation_targets.append([0, 0, 0, 1])  # Default orientation
        
        print(f"🔄 Episode reset: {len(self.active_objects)} active obstacles, goal at {goal_pos}")
    
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
