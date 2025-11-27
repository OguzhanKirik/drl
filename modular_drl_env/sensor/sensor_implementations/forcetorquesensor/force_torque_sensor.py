import numpy as np
import pybullet as pyb
from gym.spaces import Box
from typing import List, Dict
from time import process_time
from modular_drl_env.sensor.sensor import Sensor
from modular_drl_env.util.pybullet_util import pybullet_util as pybu
import copy
from time import time
from abc import abstractmethod
from modular_drl_env.robot import Robot


__all__=[
    'ForceTorqueSensor'
]


class ForceTorqueSensor(Sensor):
    """
    Base class for Force/Torque sensor for measuring forces and torques at robot joints or end-effector.
    
    This sensor can measure:
    1. Joint reaction forces (6-DOF: Fx, Fy, Fz, Mx, My, Mz) at specified joints
    2. Contact forces at the end-effector or gripper
    3. Total wrench (force + torque) at a specific link
    
    Must be subclassed for each robot to specify which joints/links to measure.
    
    Args:
        robot: Robot object to attach the sensor to
        sim_step: Simulation time step
        sim_steps_per_env_step: Number of simulation steps per environment step
        measure_joint_forces: Whether to measure joint reaction forces
        measure_contact_forces: Whether to measure contact forces at the link
        normalize: Whether to normalize output to [-1, 1]
        add_to_observation_space: Whether to add to observation space
        add_to_logging: Whether to add to logging
        update_steps: How often to update (1 = every step)
    """

    def __init__(self,
                robot: Robot,
                sim_step: float,
                sim_steps_per_env_step: int,
                measure_joint_forces: bool =True,
                measure_contact_forces: bool = True,
                force_limit:float = 100.0,
                torque_limit: float = 10.0,
                normalize:bool  = False,
                add_to_observation_space: bool = True,
                add_to_logging: bool = True,
                update_steps: int = 1,
                ):
        super().__init__(sim_step, sim_steps_per_env_step, normalize, add_to_observation_space, add_to_logging, update_steps)

        self.robot = robot

        #Configure what to measure
        self.measure_joint_forces = measure_joint_forces
        self.measure_contact_forces = measure_contact_forces

        # Normalize limites
        self.force_limit = force_limit
        self.torque_limit = torque_limit
        
        # Set output data field names
        self.output_name_joint_force = f"joint_force_{self.robot.name}"
        self.output_name_joint_torque = f"joint_torque_{self.robot.name}"
        self.output_name_contact_force = f"contact_force_{self.robot.name}"
        self.output_name_contact_torque = f"contact_torque_{self.robot.name}"
        
        # Initialize data storage
        self.joint_reaction_force = np.zeros(3)  # [Fx, Fy, Fz]
        self.joint_reaction_torque = np.zeros(3)  # [Mx, My, Mz]
        self.contact_force = np.zeros(3)  # Total contact force
        self.contact_torque = np.zeros(3)  # Total contact torque
        self.contact_normal_force = 0.0  # Normal force magnitude
        self.num_contacts = 0  # Number of contact points

    def update(self, step) -> dict:
        """Update sensor readings."""
        self.cpu_epoch = process_time()
        
        if step % self.update_steps == 0:
            # Get sensor data from the concrete implementation
            self._update_sensor_data()
        
        self.cpu_time = process_time() - self.cpu_epoch
        return self.get_observation()
    
    @abstractmethod
    def _update_sensor_data(self):
        """
        This should implement the concrete PyBullet calls to get force/torque data
        adapted to a specific robot. Must update:
        - self.joint_reaction_force
        - self.joint_reaction_torque
        - self.contact_force
        - self.contact_torque
        - self.contact_normal_force
        - self.num_contacts
        """
        pass

    def reset(self):
        """Reset sensor readings."""
        self.cpu_epoch = process_time()
        self.joint_reaction_force = np.zeros(3)
        self.joint_reaction_torque = np.zeros(3)
        self.contact_force = np.zeros(3)
        self.contact_torque = np.zeros(3)
        self.contact_normal_force = 0.0
        self.num_contacts = 0
        self.cpu_time = process_time() - self.cpu_epoch

    def get_observation(self) -> dict:
        "Get sensor observation"
        if self.normalize:
            return self._normalize()
        else:
            ret_dict = {}
            if self.measure_joint_forces:
                ret_dict[self.output_name_joint_force] = self.joint_reaction_force
                ret_dict[self.output_name_joint_torque] = self.joint_reaction_torque
            if self.measure_contact_forces:
                ret_dict[self.output_name_contact_force] = self.contact_force
                ret_dict[self.output_name_contact_torque] = self.contact_torque
            return ret_dict
        
    def _normalize(self)->dict:
        """Normalize forces and torques to [-1, 1]."""
        ret_dict = {}
        if self.measure_joint_forces:
            ret_dict[self.output_name_joint_force] = np.clip(
                self.joint_reaction_force / self.force_limit, -1.0, 1.0
            )
            ret_dict[self.output_name_joint_torque] = np.clip(
                self.joint_reaction_torque / self.torque_limit, -1.0, 1.0
            )
        if self.measure_contact_forces:
            ret_dict[self.output_name_contact_force] = np.clip(
                self.contact_force / self.force_limit, -1.0, 1.0
            )
            ret_dict[self.output_name_contact_torque] = np.clip(
                self.contact_torque / self.torque_limit, -1.0, 1.0
            )
        return ret_dict
    
    def get_observation_space_element(self)->dict:
        "Define observation space"
        if not self.add_to_observation_space:
            return {}
        
        obs_sp_ele ={}

        if self.normalize:
            #Normalized to [-1,1]
            if self.measure_joint_forces:
                obs_sp_ele[self.output_name_joint_force] = Box(low=-1, high=1, shape=(3,), dtype=np.float32)
                obs_sp_ele[self.output_name_joint_torque] = Box(low=-1, high=1, shape=(3,), dtype=np.float32)
            if self.measure_contact_forces:
                obs_sp_ele[self.output_name_contact_force] = Box(low=-1, high=1, shape=(3,), dtype=np.float32)
                obs_sp_ele[self.output_name_contact_torque] = Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        else:
            # Actual force/torque values
            if self.measure_joint_forces:
                obs_sp_ele[self.output_name_joint_force] = Box(
                    low=-self.force_limit, high=self.force_limit, shape=(3,), dtype=np.float32
                )
                obs_sp_ele[self.output_name_joint_torque] = Box(
                    low=-self.torque_limit, high=self.torque_limit, shape=(3,), dtype=np.float32
                )
            if self.measure_contact_forces:
                obs_sp_ele[self.output_name_contact_force] = Box(
                    low=-self.force_limit, high=self.force_limit, shape=(3,), dtype=np.float32
                )
                obs_sp_ele[self.output_name_contact_torque] = Box(
                    low=-self.torque_limit, high=self.torque_limit, shape=(3,), dtype=np.float32
                )
        
        return obs_sp_ele
    

    def get_data_for_logging(self) -> dict:
        """Get data for logging."""
        if not self.add_to_logging:
            return {}
        
        log_dict = {}
        log_dict[f"ft_sensor_cpu_time_{self.robot.name}"] = self.cpu_time
        
        if self.measure_joint_forces:
            log_dict[f"{self.output_name_joint_force}_x"] = float(self.joint_reaction_force[0])
            log_dict[f"{self.output_name_joint_force}_y"] = float(self.joint_reaction_force[1])
            log_dict[f"{self.output_name_joint_force}_z"] = float(self.joint_reaction_force[2])
            log_dict[f"{self.output_name_joint_torque}_x"] = float(self.joint_reaction_torque[0])
            log_dict[f"{self.output_name_joint_torque}_y"] = float(self.joint_reaction_torque[1])
            log_dict[f"{self.output_name_joint_torque}_z"] = float(self.joint_reaction_torque[2])
        
        if self.measure_contact_forces:
            log_dict[f"{self.output_name_contact_force}_x"] = float(self.contact_force[0])
            log_dict[f"{self.output_name_contact_force}_y"] = float(self.contact_force[1])
            log_dict[f"{self.output_name_contact_force}_z"] = float(self.contact_force[2])
            log_dict[f"contact_normal_force_{self.robot.name}"] = float(self.contact_normal_force)
            log_dict[f"num_contacts_{self.robot.name}"] = int(self.num_contacts)
        
        return log_dict
    
    #convenice methods for direct access
    @property
    # Convenience methods for direct access
    def get_joint_force(self) -> np.ndarray:
        """Get joint reaction force [Fx, Fy, Fz]."""
        return self.joint_reaction_force.copy()
    @property
    def get_joint_torque(self) -> np.ndarray:
        """Get joint reaction torque [Mx, My, Mz]."""
        return self.joint_reaction_torque.copy()
    @property    
    def get_contact_force(self) -> np.ndarray:
        """Get total contact force [Fx, Fy, Fz]."""
        return self.contact_force.copy()
    @property    
    def get_contact_normal_force(self) -> float:
        """Get total normal force magnitude."""
        return self.contact_normal_force
    
    @property
    def get_force_magnitude(self) -> float:
        """Get magnitude of total contact force."""
        return np.linalg.norm(self.contact_force)
    
    @property
    def is_in_contact(self) -> bool:
        """Check if sensor detects any contact."""
        return self.num_contacts > 0
