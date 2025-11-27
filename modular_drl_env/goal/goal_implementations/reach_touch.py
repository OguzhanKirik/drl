from gym.spaces import Box
import numpy as np
from modular_drl_env.goal.goal import Goal
from modular_drl_env.robot.robot import Robot
from modular_drl_env.util.pybullet_util import pybullet_util as pyb_u
import pybullet as pyb

__all__ = [
    'ReachAndTouchGoal'
]

class ReachAndTouchGoal(Goal):
    """
    Goal for reaching and touching a target object.
    Rewards the robot for getting close to the object and making contact.
    """
    
    def __init__(self, 
                 robot: Robot,
                 normalize_rewards: bool,
                 normalize_observations: bool,
                 train: bool,
                 add_to_logging: bool,
                 max_steps: int,
                 continue_after_success: bool,
                 reward_success: float = 20.0,
                 reward_contact: float = 5.0,
                 reward_distance_mult: float = -0.01,
                 reward_collision: float = -5.0,
                 dist_threshold_start: float = 0.15,
                 dist_threshold_end: float = 0.05,
                 dist_threshold_increment_start: float = 0.01,
                 dist_threshold_increment_end: float = 0.001,
                 dist_threshold_change: float = 0.8,
                 contact_threshold: float = 0.05):  # Distance considered as "contact"
        
        super().__init__(robot, normalize_rewards, normalize_observations, train, True, add_to_logging, max_steps, continue_after_success)
        
        self.robot = robot
        self.reward_success = reward_success
        self.reward_contact = reward_contact
        self.reward_distance_mult = reward_distance_mult
        self.reward_collision = reward_collision
        self.contact_threshold = contact_threshold
        
        # Distance threshold for success (curriculum learning)
        self.distance_threshold_start = dist_threshold_start
        self.distance_threshold_end = dist_threshold_end
        self.distance_threshold = dist_threshold_start
        self.distance_threshold_increment_start = dist_threshold_increment_start
        self.distance_threshold_increment_end = dist_threshold_increment_end
        self.distance_threshold_change = dist_threshold_change
        
        # State variables
        self.target_object = None
        self.target_position = None
        self.ee_position = None
        self.distance = float('inf')
        self.is_touching = False
        self.past_distances = []
        
        # Episode tracking
        self.is_success = False
        self.done = False
        self.timeout = False
        self.collided = False
        self.out_of_bounds = False
        self.reward_value = 0
        
        # Output name for observations
        self.output_name = "target_relative_position_" + self.robot.name
        
        # Normalizing constants
        max_workspace_dist = np.linalg.norm([
            self.robot.world.x_max - self.robot.world.x_min,
            self.robot.world.y_max - self.robot.world.y_min,
            self.robot.world.z_max - self.robot.world.z_min
        ])
        self.normalizing_constant_a_obs = 2.0 / max_workspace_dist
        self.normalizing_constant_b_obs = -1.0

    def get_observation_space_element(self) -> dict:
        if self.add_to_observation_space:
            ret = dict()
            if self.normalize_observations:
                ret[self.output_name] = Box(low=-1, high=1, shape=(4,), dtype=np.float32)
            else:
                high = np.array([10.0, 10.0, 10.0, 10.0], dtype=np.float32)
                low = np.array([-10.0, -10.0, -10.0, 0.0], dtype=np.float32)
                ret[self.output_name] = Box(low=low, high=high, shape=(4,), dtype=np.float32)
            return ret
        else:
            return {}

    def get_observation(self) -> dict:
        # Get end-effector position
        self.ee_position = self.robot.position_rotation_sensor.position
        
        # Get target object position from world
        if len(self.robot.world.position_targets) > 0:
            self.target_position = self.robot.world.position_targets[self.robot.mgt_id]
        else:
            self.target_position = np.array([0.0, 0.0, 0.0])
        
        # Calculate relative position and distance
        dif = self.target_position - self.ee_position
        self.distance = np.linalg.norm(dif)
        
        # Track distance history
        self.past_distances.append(self.distance)
        if len(self.past_distances) > 10:
            self.past_distances.pop(0)
        
        # Build observation: [dx, dy, dz, distance]
        ret = np.zeros(4, dtype=np.float32)
        ret[:3] = dif
        ret[3] = self.distance
        
        if self.normalize_observations:
            ret = self.normalizing_constant_a_obs * ret + self.normalizing_constant_b_obs
            return {self.output_name: ret}
        else:
            return {self.output_name: ret}

    def reward(self, step, action):
        reward = 0
        
        # Get current state
        self.out_of_bounds = self.robot.world.out_of_bounds(self.ee_position)
        self.collided = pyb_u.collision
        
        # Check if touching the target object
        self.is_touching = self.distance < self.contact_threshold
        
        # Shaking penalty (penalize jittery movement)
        shaking = 0
        if len(self.past_distances) >= 10:
            arrow = []
            for i in range(9):
                arrow.append(0 if self.past_distances[i + 1] - self.past_distances[i] >= 0 else 1)
            for j in range(8):
                if arrow[j] != arrow[j+1]:
                    shaking += 1
        reward -= shaking * 0.005
        
        # Determine episode outcome
        self.is_success = False
        
        if self.out_of_bounds:
            self.done = True
            reward += self.reward_collision
        elif self.collided:
            self.done = True
            reward += self.reward_collision
        elif self.distance < self.distance_threshold and self.is_touching:
            # Success: reached and touching the object
            self.done = True
            self.is_success = True
            reward += self.reward_success
        elif step > self.max_steps:
            self.done = True
            self.timeout = True
            reward += self.reward_collision / 10
        else:
            self.done = False
            # Dense reward shaping
            reward += self.reward_distance_mult * self.distance
            # Extra reward for getting very close (contact zone)
            if self.is_touching:
                reward += self.reward_contact
        
        self.reward_value = reward
        
        return self.reward_value, self.is_success, self.done, self.timeout, self.out_of_bounds

    def on_env_reset(self, success_rate):
        # Reset episode state
        self.timeout = False
        self.is_success = False
        self.done = False
        self.collided = False
        self.out_of_bounds = False
        self.is_touching = False
        self.past_distances = []
        
        # Curriculum learning: adjust distance threshold based on success rate
        if self.train:
            ratio_start_end = (self.distance_threshold - self.distance_threshold_end) / (self.distance_threshold_start - self.distance_threshold_end)
            increment = (self.distance_threshold_increment_start - self.distance_threshold_increment_end) * ratio_start_end + self.distance_threshold_increment_end
            
            if success_rate > self.distance_threshold_change and self.distance_threshold > self.distance_threshold_end:
                self.distance_threshold -= increment
            elif success_rate < self.distance_threshold_change and self.distance_threshold < self.distance_threshold_start:
                pass  # Don't increase threshold
            
            if self.distance_threshold > self.distance_threshold_start:
                self.distance_threshold = self.distance_threshold_start
            if self.distance_threshold < self.distance_threshold_end:
                self.distance_threshold = self.distance_threshold_end
        
        return [("distance_threshold", self.distance_threshold, True, True)]

    def build_visual_aux(self):
        # Draw a sphere around the target object
        if len(self.robot.world.position_targets) > 0:
            target_pos = self.robot.world.position_targets[self.robot.mgt_id]
            self.aux_object_ids.append(
                pyb_u.create_sphere(
                    position=target_pos, 
                    mass=0, 
                    radius=self.distance_threshold, 
                    color=[0, 1, 0, 0.3],  # Green transparent
                    collision=False
                )
            )

    def get_data_for_logging(self) -> dict:
        logging_dict = dict()
        logging_dict["reward_" + self.robot.name] = self.reward_value
        logging_dict["distance_to_target_" + self.robot.name] = self.distance
        logging_dict["distance_threshold_" + self.robot.name] = self.distance_threshold
        logging_dict["is_touching_" + self.robot.name] = int(self.is_touching)
        return logging_dict
