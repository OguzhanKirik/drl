from ..force_torque_sensor import ForceTorqueSensor
from modular_drl_env.robot.robot import Robot
import numpy as np
import pybullet as pyb
from modular_drl_env.util.pybullet_util import pybullet_util as pyb_u

__all__ =[
    'ForceTorqueSensorUR5'
]

class ForceTorqueSensorUR5(ForceTorqueSensor):
    """
    Force/Torque sensor implementation for UR5 robot.
    
    Measures forces and torques at the wrist joint and contact forces at the end-effector.
    Automatically configures for UR5 robot structure.
    
    Args:
        robot: UR5 Robot object
        sim_step: Simulation time step
        sim_steps_per_env_step: Number of simulation steps per environment step
        joint_name: Which joint to measure forces at ('wrist_3', 'wrist_2', 'wrist_1', etc.)
        measure_joint_forces: Whether to measure joint reaction forces
        measure_contact_forces: Whether to measure contact forces at end-effector
        force_limit: Max force in Newtons for normalization
        torque_limit: Max torque in Nm for normalization
        normalize: Whether to normalize output to [-1, 1]
        add_to_observation_space: Whether to add to observation space
        add_to_logging: Whether to add to logging
        update_steps: How often to update (1 = every step)
    """


    def __init__(self, 
                 robot: Robot,
                 sim_step: float, 
                 sim_steps_per_env_step: int,
                 joint_name: str = 'wrist_3_link',  # Default to wrist 3
                 measure_joint_forces: bool = True,
                 measure_contact_forces: bool = True,
                 force_limit: float = 100.0,
                 torque_limit: float = 10.0,
                 normalize: bool = False, 
                 add_to_observation_space: bool = True, 
                 add_to_logging: bool = True, 
                 update_steps: int = 1
                 ):
        super().__init__(
            robot=robot,
            sim_step=sim_step,
            sim_steps_per_env_step=sim_steps_per_env_step,
            measure_joint_forces=measure_joint_forces,
            measure_contact_forces=measure_contact_forces,
            force_limit=force_limit,
            torque_limit=torque_limit,
            normalize=normalize,
            add_to_observation_space=add_to_observation_space,
            add_to_logging=add_to_logging,
            update_steps=update_steps
        )

        # UR5 specific config
        # UR5-specific configuration
        self.joint_name = joint_name
        self.joint_index = None
        self.link_index = None
        self.robot_pyb_id = None
        self.torque_sensor_enabled = False

    def _update_sensor_data(self):
        """Get force/torque data from UR5 robot."""
        #Initialize on first call
        if self.robot_pyb_id is None:
            self.robot_pyb_id= pyb_u.to_pb(self.robot.object_id)
            self._find_joint_and_link_indices()
            self._enable_torque_sensor()

        # Measure joint reaction forces
        if self.measure_joint_forces and self.joint_index is not None:
            joint_state = pyb.getJointState(self.robot_pyb_id, self.joint_index)
            # joint_state[2] contains [Fx, Fy, Fz, Mx, My, Mz]
            reaction_wrench = np.array(joint_state[2])
            self.joint_reaction_force = reaction_wrench[0:3]
            self.joint_reaction_torque = reaction_wrench[3:6]

        # Measure contact forces at end-effector
        if self.measure_contact_forces and self.link_index is not None:
            self._update_contact_forces()

    def _find_joint_and_link_indices(self):
        """find the joint and link indices for UR5"""
        nun_joints = pyb.getNumJoints(self.robot_pyb_id)

        for i in range(nun_joints):
            joint_info = pyb.getJointInfo(self.robot_pyb_id,i)
            link_name = joint_info[12].decode('utf-8')
            joint_name = joint_info[1].decode('utf-8')
            
            # find the joint the measure forces at
            if link_name == self.joint_name:
                self.joint_index = i

            # Find end effector link for contact forces
            if link_name == self.robot.end_effector_link_id:
                self.link_index = i 


    def _enable_torque_sensor(self):
        """Enable the joint torque sensor in PyBullet."""
        if not self.torque_sensor_enabled and self.measure_joint_forces and self.joint_index is not None:
            pyb.enableJointForceTorqueSensor(
                bodyUniqueId=self.robot_pyb_id,
                jointIndex=self.joint_index,
                enableSensor=True
            )
            self.torque_sensor_enabled = True

    def _update_contact_forces(self):
        """Calculate total contact forces at the end-effector."""
        # Get all contact points on the end-effector link
        contact_points = pyb.getContactPoints(bodyA=self.robot_pyb_id, linkIndexA=self.link_index)
    
        self.num_contacts = len(contact_points)

        if self.num_contacts > 0:
            # Accumulate forces from all contact points
            total_force = np.zeros(3)
            total_normal_force = 0.0

            for contact in contact_points:
                #contract[9] = normal force magnitdude
                normal_force = contact[9]
                total_normal_force += normal_force
                
                # contact[7] = contact normal direction (from bodyA to bodyB)
                contact_normal = np.array(contact[7])
                
                # contact[10] = lateral friction force 1
                # contact[11] = lateral friction direction 1
                # contact[12] = lateral friction force 2
                # contact[13] = lateral friction direction 2
                
                # Total force = normal force + friction forces
                force_normal = normal_force * contact_normal
                
                lateral_force_1 = contact[10] * np.array(contact[11])
                lateral_force_2 = contact[12] * np.array(contact[13])
                
                total_force += force_normal + lateral_force_1 + lateral_force_2
                    
            self.contact_force = total_force
            self.contact_normal_force = total_normal_force
            
            # For torque, we would need to calculate moment arms
            # This is simplified - could be extended for accurate torque calculation
            self.contact_torque = np.zeros(3)  # Placeholder
        else:
            self.contact_force = np.zeros(3)
            self.contact_torque = np.zeros(3)
            self.contact_normal_force = 0.0
