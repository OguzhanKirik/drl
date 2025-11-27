"""
Grasp Object Goal

This goal rewards the robot for grasping and lifting target objects.
Success is defined as:
1. Gripper is closed (gripper_state > threshold)
2. Object is grasped (constraint exists)
3. Object is lifted above a minimum height
"""

import numpy as np
from gym.spaces import Box
from modular_drl_env.goal.goal import Goal
from modular_drl_env.robot.robot import Robot
from modular_drl_env.util.pybullet_util import pybullet_util as pyb_u
import pybullet as pyb


class GraspObjectGoal(Goal):
    """
    Goal that rewards grasping and lifting objects.
    
    Reward structure:
    - Distance shaping: Encourages moving towards target
    - Contact bonus: Reward for being near target
    - Grasp bonus: Reward for successfully grasping
    - Lift bonus: Reward for lifting object
    - Success reward: Large reward for successful grasp+lift
    - Collision penalty: Penalty for hitting obstacles
    """
    
    def __init__(self,
                 robot: Robot,
                 normalize_rewards: bool,
                 normalize_observations: bool,
                 train: bool,
                 add_to_logging: bool,
                 max_steps: int,
                 continue_after_success: bool,
                 reward_success: float = 50.0,
                 reward_lift: float = 10.0,
                 reward_grasp: float = 5.0,
                 reward_contact: float = 2.0,
                 reward_collision: float = -5.0,
                 reward_distance_mult: float = -0.01,
                 contact_threshold: float = 0.08,
                 lift_threshold: float = 0.15,
                 dist_threshold_start: float = 0.10,
                 dist_threshold_end: float = 0.03,
                 dist_threshold_increment_start: float = 0.005,
                 dist_threshold_increment_end: float = 0.001,
                 dist_threshold_change: float = 0.7):
        
        super().__init__(robot, normalize_rewards, normalize_observations, train, True, add_to_logging, max_steps, continue_after_success)
        
        # Reward parameters
        self.reward_success = reward_success
        self.reward_lift = reward_lift
        self.reward_grasp = reward_grasp
        self.reward_contact = reward_contact
        self.reward_collision = reward_collision
        self.reward_distance_mult = reward_distance_mult
        
        # Thresholds
        self.contact_threshold = contact_threshold
        self.lift_threshold = lift_threshold
        
        # Curriculum learning for distance threshold
        self.dist_threshold = dist_threshold_start
        self.dist_threshold_start = dist_threshold_start
        self.dist_threshold_end = dist_threshold_end
        self.dist_threshold_increment_start = dist_threshold_increment_start
        self.dist_threshold_increment_end = dist_threshold_increment_end
        self.dist_threshold_change = dist_threshold_change
        
        # State tracking
        self.target_position = None
        self.target_initial_height = None
        self.previous_distance = None
        self.success_count = 0
        self.total_count = 0
        
        # Episode state
        self.is_success = False
        self.done = False
        self.timeout = False
        self.out_of_bounds = False
        self.reward_value = 0.0
        
        # Output name for observations
        self.output_name = "grasp_observation_" + self.robot.name
        
        # Metric names for curriculum learning
        self.metric_names = ["distance_threshold"]
        
        # Normalizing constants
        max_workspace_dist = np.linalg.norm([
            self.robot.world.x_max - self.robot.world.x_min,
            self.robot.world.y_max - self.robot.world.y_min,
            self.robot.world.z_max - self.robot.world.z_min
        ])
        self.normalizing_constant_obs = 2.0 / max_workspace_dist
        
    def get_observation_space_element(self) -> dict:
        """Define the observation space for this goal."""
        if self.add_to_observation_space:
            ret = dict()
            if self.normalize_observations:
                ret[self.output_name] = Box(low=-1, high=1, shape=(7,), dtype=np.float32)
            else:
                # [dx, dy, dz, distance, gripper_state, is_grasped, height_delta]
                high = np.array([10.0, 10.0, 10.0, 10.0, 1.0, 1.0, 2.0], dtype=np.float32)
                low = np.array([-10.0, -10.0, -10.0, 0.0, 0.0, 0.0, -2.0], dtype=np.float32)
                ret[self.output_name] = Box(low=low, high=high, shape=(7,), dtype=np.float32)
            return ret
        else:
            return {}
    
    def get_observation(self) -> dict:
        """
        Returns observation dict with:
        - Combined vector: [dx, dy, dz, distance, gripper_state, is_grasped, height_delta]
        """
        ee_pos = np.array(self.robot.position_rotation_sensor.position)
        
        # Get current target position
        if hasattr(self.robot.world, 'target_object') and self.robot.world.target_object is not None:
            obj_id = pyb_u.to_pb(self.robot.world.target_object.object_id)
            obj_pos, _ = pyb.getBasePositionAndOrientation(obj_id)
            current_target_pos = np.array(obj_pos)
        else:
            current_target_pos = self.target_position if self.target_position is not None else np.zeros(3)
        
        # Relative position to target
        relative_pos = current_target_pos - ee_pos
        distance = np.linalg.norm(relative_pos)
        
        # Gripper state
        gripper_state = self.robot.gripper_state if hasattr(self.robot, 'gripper_state') else 0.0
        
        # Grasp state
        is_grasped = 1.0 if (hasattr(self.robot, 'grasp_constraint') and 
                            self.robot.grasp_constraint is not None) else 0.0
        
        # Height delta
        height_delta = current_target_pos[2] - self.target_initial_height if self.target_initial_height is not None else 0.0
        
        # Combine into single observation vector
        obs_vector = np.concatenate([
            relative_pos,
            [distance, gripper_state, is_grasped, height_delta]
        ]).astype(np.float32)
        
        # Normalize if requested
        if self.normalize_observations:
            obs_vector[:3] = obs_vector[:3] * self.normalizing_constant_obs  # normalize position
            obs_vector[3] = obs_vector[3] * self.normalizing_constant_obs  # normalize distance
            # gripper_state, is_grasped already in [0, 1]
            obs_vector[6] = np.clip(obs_vector[6] / 0.5, -1, 1)  # normalize height_delta to [-1, 1]
        
        obs = {self.output_name: obs_vector}
        
        return obs
    
    def reward(self, step, action):
        """
        Computes reward based on:
        - Distance to target (shaping)
        - Contact with target
        - Grasping target
        - Lifting target
        - Success (grasp + lift)
        
        Returns: (reward, success, done, timeout, out_of_bounds)
        """
        reward_val = 0.0
        
        # Get end-effector position
        ee_pos = np.array(self.robot.position_rotation_sensor.position)
        
        # Update target position if object is grasped (it moves with gripper)
        if hasattr(self.robot.world, 'target_object') and self.robot.world.target_object is not None:
            obj_id = pyb_u.to_pb(self.robot.world.target_object.object_id)
            obj_pos, _ = pyb.getBasePositionAndOrientation(obj_id)
            current_target_pos = np.array(obj_pos)
        else:
            current_target_pos = self.target_position
        
        # Calculate distance to target
        distance = np.linalg.norm(ee_pos - current_target_pos)
        
        # Distance shaping reward (encourages moving towards target)
        if self.previous_distance is not None:
            delta_distance = self.previous_distance - distance
            reward_val += delta_distance * abs(self.reward_distance_mult) * 10  # Scale up shaping
        reward_val += self.reward_distance_mult * distance
        self.previous_distance = distance
        
        # Contact bonus (close to target)
        if distance < self.contact_threshold:
            reward_val += self.reward_contact
        
        # Grasp bonus (gripper closed and holding object)
        is_grasped = False
        if hasattr(self.robot, 'grasp_constraint') and self.robot.grasp_constraint is not None:
            is_grasped = True
            reward_val += self.reward_grasp
        
        # Lift bonus (object lifted above initial height)
        is_lifted = False
        if is_grasped and self.target_initial_height is not None:
            height_delta = current_target_pos[2] - self.target_initial_height
            if height_delta > self.lift_threshold:
                is_lifted = True
                reward_val += self.reward_lift
        
        # Success reward (grasped AND lifted)
        self.is_success = is_grasped and is_lifted
        if self.is_success:
            reward_val += self.reward_success
            self.success_count += 1
        
        # Collision penalty
        if len(self.robot.sensors) > 0 and hasattr(self.robot.sensors[0], 'collision_occurred'):
            if self.robot.sensors[0].collision_occurred:
                reward_val += self.reward_collision
        
        # Check termination conditions
        self.timeout = (step >= self.max_steps)
        self.out_of_bounds = False
        if len(self.robot.sensors) > 0 and hasattr(self.robot.sensors[0], 'out_of_bounds'):
            self.out_of_bounds = self.robot.sensors[0].out_of_bounds
        self.done = self.is_success and not self.continue_after_success
        
        # Normalize reward if requested
        if self.normalize_rewards:
            reward_val = reward_val / (abs(self.reward_success) + abs(self.reward_lift) + abs(self.reward_grasp))
        
        self.reward_value = reward_val
        
        return reward_val, self.is_success, self.done, self.timeout, self.out_of_bounds
    
    def on_env_reset(self, success_rate):
        """
        Called at the start of each episode.
        Gets target object position from world and updates curriculum.
        
        Returns: list of tuples (metric_name, value, can_write_back, lower_is_better)
        """
        # Get target position from world
        if len(self.robot.world.position_targets) > 0:
            self.target_position = np.array(self.robot.world.position_targets[0])
            self.target_initial_height = self.target_position[2]
        else:
            # Fallback if no targets yet
            self.target_position = np.array([0.0, 0.5, 1.0])
            self.target_initial_height = 1.0
        
        # Reset episode state
        self.previous_distance = None
        self.is_success = False
        self.done = False
        self.timeout = False
        self.out_of_bounds = False
        self.reward_value = 0.0
        self.total_count += 1
        
        # Update curriculum learning
        if self.total_count > 10 and success_rate >= self.dist_threshold_change:
            # Gradually reduce threshold
            increment = self.dist_threshold_increment_start + \
                       (self.dist_threshold_increment_end - self.dist_threshold_increment_start) * \
                       (self.dist_threshold_start - self.dist_threshold) / \
                       (self.dist_threshold_start - self.dist_threshold_end + 1e-6)
            
            old_threshold = self.dist_threshold
            self.dist_threshold = max(
                self.dist_threshold_end,
                self.dist_threshold - increment
            )
            
            if old_threshold != self.dist_threshold:
                print(f"[GraspObjectGoal] Curriculum updated: distance_threshold={self.dist_threshold:.4f}")
        
        # Return metrics (name, value, can_write_back, lower_is_better)
        return [("distance_threshold", self.dist_threshold, True, True)]


__all__ = ['GraspObjectGoal']
