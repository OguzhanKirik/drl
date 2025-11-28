#!/usr/bin/env python3
"""
Test script for the Force/Torque sensor.
Shows how to use the UR5-specific sensor implementation.
"""

import sys
import time
import cv2
import numpy as np
import pybullet as pyb

# Add project to path
sys.path.insert(0, '.')

from modular_drl_env.util.pybullet_util import pybullet_util as pyb_u
from modular_drl_env.robot.robot_implementations.ur5 import UR5_Gripper
from modular_drl_env.sensor.sensor_implementations.forcetorquesensor import ForceTorqueSensorUR5
from modular_drl_env.sensor.sensor_implementations.positional import PositionRotationSensor, JointsSensor
from modular_drl_env.world.obstacles.shapes import Box, Sphere, Cylinder
from modular_drl_env.sensor.sensor_implementations.camera.camera_implementations.camera_implementations import (
    StaticBodyCameraUR5, 
    StaticFloatingCamera
)

class SimpleWorld:
    """Minimal world for testing."""
    def __init__(self):
        self.x_min, self.x_max = -1.0, 1.0
        self.y_min, self.y_max = -1.0, 1.0
        self.z_min, self.z_max = 0.0, 2.0
        self.objects = []
        self.position_targets = []

def main():
    print("=" * 60)
    print("Force/Torque Sensor Test Script (UR5)")
    print("=" * 60)
    
    # Connect to PyBullet with GUI
    pyb_u.init(
        assets_path="./modular_drl_env/assets",
        display_mode=True,
        sim_step=0.01,
        gravity=[0, 0, -9.8]
    )
    
    # Improve physics parameters to reduce penetration
    pyb.setPhysicsEngineParameter(
        numSolverIterations=150,  # Increase from default 50
        numSubSteps=4,            # More sub-steps for better collision
        contactBreakingThreshold=0.001,  # Smaller threshold
        erp=0.8,                  # Error reduction parameter (0-1)
        contactERP=0.8,           # Contact error reduction
        frictionERP=0.8,          # Friction error reduction
        solverResidualThreshold=1e-7  # Better solver convergence
    )
    
    # Set camera
    pyb.resetDebugVisualizerCamera(
        cameraDistance=1.5,
        cameraYaw=45,
        cameraPitch=-30,
        cameraTargetPosition=[0.3, 0.0, 0.8]
    )
    
    # Load ground plane
    pyb_u.add_ground_plane(position=np.array([0, 0, 0]))
    
    # Create simple world
    world = SimpleWorld()
    
    # Create a test box with mass (so it can be moved)
    # Place it on the ground - small box for testing
    box_half_extent = 0.02  # 2cm half-extent = 4cm cube
    test_box = Box(
        position=[-0.3, 0.3, box_half_extent],  # z = halfExtent so box sits on ground
        rotation=[0, 0, 0],
        trajectory=[],
        sim_step=0.01,
        sim_steps_per_env_step=1,
        velocity=0,
        halfExtents=[box_half_extent, box_half_extent, box_half_extent],
        color=[1, 0, 0, 1],
        seen_by_obstacle_sensor=True,
        mass=0.05  # Add mass to make box dynamic (movable)
    )
    test_box.build()
    world.objects.append(test_box)


    sphere_radius = 0.02  # 2cm radius sphere
    test_sphere = Sphere(
        position=[0.3, 0.3, sphere_radius],  # z = radius so sphere sits on ground
        trajectory=[],
        sim_step=0.01,
        sim_steps_per_env_step=1,
        velocity=0,
        radius=sphere_radius,
        color=[0, 1, 0, 1],  # Green sphere
        seen_by_obstacle_sensor=True,
        mass=0.05  # Add mass to make sphere dynamic (movable)
    )
    test_sphere.build()
    world.objects.append(test_sphere)


    cylinder_radius = 0.02  # 2cm radius
    cylinder_height = 0.08  # 8cm height
    test_cylinder = Cylinder(
        position=[0.0, 0.3, cylinder_height/2],  # z = height/2 so cylinder sits on ground
        rotation=[0, 0, 0],
        trajectory=[],
        sim_step=0.01,
        sim_steps_per_env_step=1,
        velocity=0,
        radius=cylinder_radius,
        height=cylinder_height,
        color=[0, 0, 1, 1],  # Blue cylinder
        seen_by_obstacle_sensor=True,
        mass=0.05  # Add mass to make cylinder dynamic (movable)
    )
    test_cylinder.build()
    world.objects.append(test_cylinder)


    print(f"✅ Created test box at position: {test_box.position}, mass: 0.05 kg, size: 4cm cube")
    
    # Create UR5 robot with gripper
    robot = UR5_Gripper(
        name="ur5_robot",
        id_num=0,
        world=world,
        sim_step=0.01,
        use_physics_sim=True,
        base_position=[0, 0, 0.0],
        base_orientation=[0, 0, 0, 1],
        resting_angles=np.array([0, -1.57, -1.57, -1.57, 1.57, 0, 0.015, 0.015]),
        control_mode=0,  # Use IK delta control mode
        ik_xyz_delta=0.05,
        ik_rpy_delta=0.05,
        gripper_control=True
    )
    robot.build()
    
    print(f"✅ Created UR5_Gripper robot")
    
    # Create Force/Torque Sensor for UR5
    ft_sensor = ForceTorqueSensorUR5(
        robot=robot,
        sim_step=0.01,
        sim_steps_per_env_step=1,
        joint_name='wrist_3_link',  # Measure at wrist 3
        measure_joint_forces=True,
        measure_contact_forces=True,
        force_limit=100.0,
        torque_limit=10.0,
        normalize=False,
        add_to_observation_space=True,
        add_to_logging=True,
        update_steps=1
    )
    
    print(f"✅ Created Force/Torque Sensor (UR5-specific)")
    print(f"   Measuring at joint: {ft_sensor.joint_name}")
    print(f"   Force limit: {ft_sensor.force_limit}N")
    print(f"   Torque limit: {ft_sensor.torque_limit}Nm")
    
    # Create Camera Sensors
    # Camera 1: Mounted on robot end-effector (moves with robot)
    camera_args = {
        'type': 'rgb',  # Options: 'rgb', 'rgbd', 'grayscale'
        'width': 1920,
        'height': 1080,
        'fov': 60,
        'aspect': 1,
        'near_val': 0.05,
        'far_val': 2,  # Must be int to match default type
        'up_vector': [0, 0, 1]
    }
    
    ee_camera = StaticBodyCameraUR5(
        robot=robot,
        position_relative_to_effector=[0.0, 0.0, 0.05],  # 5cm offset from end-effector
        camera_args=camera_args,
        name='end_effector_camera',
        normalize=False,
        add_to_observation_space=False,  # Set True if you want it in observations
        add_to_logging=False,
        sim_step=0.01,
        sim_steps_per_env_step=1,
        update_steps=1
    )
    
    # Camera 2: Static overhead camera (tilted for better view)
    static_camera = StaticFloatingCamera(
        position=[0.0, 1.0, 1.0],  # Side angle: 0.5m right, 0.5m forward, 0.8m up
        target=[0.0, 0.3, 0.1],    # Looking at object area (slightly above ground)
        camera_args=camera_args,
        name='overhead_camera',
        normalize=False,
        add_to_observation_space=False,
        add_to_logging=False,
        sim_step=0.01,
        sim_steps_per_env_step=1,
        update_steps=1
    )
    
    print(f"✅ Created Camera Sensors")
    print(f"   End-effector camera: {camera_args['type']} {camera_args['width']}x{camera_args['height']}")
    print(f"   Overhead camera: Position [0.5, 0.5, 0.8] → Target [0.0, 0.3, 0.1] (tilted side view)")
    print(f"   Overhead camera: Static at [0, 0, 1.0]")
    
    print(f"✅ Created Force/Torque Sensor (UR5-specific)")
    print(f"   Measuring at joint: {ft_sensor.joint_name}")
    print(f"   Force limit: {ft_sensor.force_limit}N")
    print(f"   Torque limit: {ft_sensor.torque_limit}Nm")
    
    # Get PyBullet IDs
    robot_pyb_id = pyb_u.to_pb(robot.object_id)
    box_pyb_id = pyb_u.to_pb(test_box.object_id)
    
    # Find end-effector link
    ee_link_index = -1
    for i in range(pyb.getNumJoints(robot_pyb_id)):
        link_info = pyb.getJointInfo(robot_pyb_id, i)
        link_name = link_info[12].decode('utf-8')
        if link_name == robot.end_effector_link_id:
            ee_link_index = i
            break
    
    print(f"   End-effector link index: {ee_link_index}")
    
    # Create and attach real sensors to robot
    position_rotation_sensor = PositionRotationSensor(
        robot=robot,
        link_id=robot.end_effector_link_id,  # Use string ID, not integer index
        sim_step=0.01,
        sim_steps_per_env_step=1,
        quaternion=True,
        normalize=False,
        add_to_observation_space=False,  # Not needed for this test
        add_to_logging=False
    )
    
    joints_sensor = JointsSensor(
        robot=robot,
        sim_step=0.01,
        sim_steps_per_env_step=1,
        add_joint_velocities=False,
        normalize=False,
        add_to_observation_space=False,  # Not needed for this test
        add_to_logging=False
    )
    
    # Attach sensors to robot
    robot.position_rotation_sensor = position_rotation_sensor
    robot.joints_sensor = joints_sensor
    robot.sensors = [position_rotation_sensor, joints_sensor]
    
    # Initialize sensors
    position_rotation_sensor.reset()
    joints_sensor.reset()
    
    # Initialize force/torque sensor
    ft_sensor.reset()
    
    # Initialize cameras
    ee_camera.reset()
    static_camera.reset()
    
    print("\n📸 Camera test - capturing initial images...")
    ee_camera.update(0)
    static_camera.update(0)
    static_image = static_camera.get_observation()[f'camera_rgb_overhead_camera']
    print(f"   Overhead camera image shape: {static_image.shape}")
    
    print("Save Image")
    cv2.imwrite('overhead_camera_initial.png', cv2.cvtColor(static_image, cv2.COLOR_RGB2BGR))
    
    # Test 1: Measure forces while approaching box
    
    print("\n" + "=" * 60)
    print("Test 1: Approaching Box")
    print("=" * 60)
    
    box_pos = np.array(test_box.position)
    # Position gripper at box center height for horizontal grasp
    # Box is at z=0.025 (center), halfExtent=0.025
    # We want the gripper fingers to close horizontally at box center height
    target_pos = box_pos.copy()
    # Don't raise Z - keep at box center so fingers grasp horizontally
    # target_pos[2] is already at box center (0.025m)
    
    # Open gripper
    print("\n[Step 1] Opening gripper...")
    for i in range(30):
        action = np.array([0, 0, 0, 0, 0, 0, -1.0])
        robot.process_action(action)
        pyb.stepSimulation()
        position_rotation_sensor.update(i)
        ft_sensor.update(i)
        time.sleep(0.02)
    
    # Move to box
    print("\n[Step 2] Moving towards box...")
    for i in range(500):  # Increase steps to ensure we reach the box
        # Update sensors first
        position_rotation_sensor.update(i)
        
        current_pos = position_rotation_sensor.position
        delta = target_pos - current_pos
        distance = np.linalg.norm(delta)
        
        if distance < 0.005:  # Closer threshold
            print(f"   ✅ Reached target! Distance={distance:.4f}m")
            
            # Capture image from wrist camera when target is reached
            print("\n📸 Capturing wrist camera image at target...")
            ee_camera.update(i)
            ee_image_at_target = ee_camera.get_observation()[f'camera_rgb_end_effector_camera']
            cv2.imwrite('ee_camera_at_target.png', cv2.cvtColor(ee_image_at_target, cv2.COLOR_RGB2BGR))
            print(f"   Saved: ee_camera_at_target.png (shape: {ee_image_at_target.shape})")
            
            break
        
        # Calculate incremental movement action
        # Normalize the delta and scale it appropriately
        # Action values should be small (-1 to 1) since they're multiplied by ik_xyz_delta (0.05m)
        if distance > 0.02:
            # Far from target - moderate speed
            movement = (delta / distance) * 0.5  # 0.5 * 0.05 = 2.5cm per step
        else:
            # Close to target - slow for precision
            movement = (delta / distance) * 0.2  # 0.2 * 0.05 = 1cm per step
        
        # Create action: [dx, dy, dz, droll, dpitch, dyaw, gripper]
        action = np.array([
            movement[0],
            movement[1], 
            movement[2],
            0, 0, 0,  # No rotation change
            -1.0      # Gripper open
        ])
        
        robot.process_action(action)
        pyb.stepSimulation()
        ft_sensor.update(i)
        
        # Show force readings every 50 steps
        if i % 50 == 0:
            joint_force = ft_sensor.get_joint_force
            contact_force = ft_sensor.get_contact_force
            print(f"   Step {i}: Distance={distance:.4f}m, Pos: [{current_pos[0]:.3f}, {current_pos[1]:.3f}, {current_pos[2]:.3f}]")
            print(f"      Joint Force: [{joint_force[0]:.2f}, {joint_force[1]:.2f}, {joint_force[2]:.2f}] N")
            print(f"      Contact Force: [{contact_force[0]:.2f}, {contact_force[1]:.2f}, {contact_force[2]:.2f}] N")
            print(f"      In contact: {ft_sensor.is_in_contact}")
        
        time.sleep(0.01)
    


    # Test 2: Close gripper and measure grasping forces
    print("\n" + "=" * 60)
    print("Test 2: Grasping Box - Measuring Forces")
    print("=" * 60)
    
    print("\n[Step 3] Closing gripper...")
    
    # Get box PyBullet ID for contact checking
    box_pyb_id = pyb_u.to_pb(test_box.object_id)
    
    for i in range(100):
        # Close gripper using robot's process_action (no arm movement)
        action = np.array([
            0.0,  # dx
            0.0,  # dy
            0.0,  # dz
            0.0,  # roll
            0.0,  # pitch
            0.0,  # yaw
            1.0   # gripper: 1.0 = fully closed
        ])
        
        robot.process_action(action)
        pyb.stepSimulation()
        ft_sensor.update(i)
        
        # Show forces during grasping
        if i % 20 == 0:
            # Get box current position
            box_state = pyb.getBasePositionAndOrientation(box_pyb_id)
            box_current_pos = box_state[0]
            
            # Get gripper finger positions
            left_finger_state = pyb.getLinkState(robot_pyb_id, 7)  # rg2_leftfinger
            right_finger_state = pyb.getLinkState(robot_pyb_id, 8)  # rg2_rightfinger
            left_finger_pos = left_finger_state[0]
            right_finger_pos = right_finger_state[0]
            
            # Calculate distance between fingers
            finger_distance = abs(left_finger_pos[1] - right_finger_pos[1])
            
            # Check for contacts between robot and box
            contact_points = pyb.getContactPoints(bodyA=robot_pyb_id, bodyB=box_pyb_id)
            
            # Check for penetration depth
            max_penetration = 0
            if len(contact_points) > 0:
                max_penetration = max([abs(cp[8]) for cp in contact_points])
            
            joint_force = ft_sensor.get_joint_force
            contact_force = ft_sensor.get_contact_force
            normal_force = ft_sensor.get_contact_normal_force
            print(f"   Step {i}:")
            print(f"      Box position: [{box_current_pos[0]:.3f}, {box_current_pos[1]:.3f}, {box_current_pos[2]:.3f}]")
            print(f"      Finger distance: {finger_distance*1000:.1f}mm (box width: {box_half_extent*2*1000:.1f}mm)")
            print(f"      Physical contacts: {len(contact_points)}, Max penetration: {max_penetration*1000:.2f}mm")
            print(f"      Joint Force: [{joint_force[0]:.2f}, {joint_force[1]:.2f}, {joint_force[2]:.2f}] N")
            print(f"      Contact Force: [{contact_force[0]:.2f}, {contact_force[1]:.2f}, {contact_force[2]:.2f}] N")
            print(f"      Normal Force: {normal_force:.2f} N")
            print(f"      Num Contacts: {ft_sensor.num_contacts}")
        
        time.sleep(0.03)
    
    print(f"\n✅ Grasp complete!")
    print(f"   Final contact force magnitude: {ft_sensor.get_force_magnitude:.2f} N")
    print(f"   Grasped: {robot.grasp_constraint is not None}")
    if robot.grasp_constraint is not None:
        print(f"   Constraint ID: {robot.grasp_constraint}")
        # Get constraint info
        constraint_info = pyb.getConstraintInfo(robot.grasp_constraint)
        print(f"   Constraint details: Parent={constraint_info[2]}, Child={constraint_info[3]}")
    
    # Test 3: Lift and monitor forces
    print("\n" + "=" * 60)
    print("Test 3: Lifting Box - Force Monitoring")
    print("=" * 60)
    
    input("\nPress ENTER to lift...")
    
    # Check constraint before lifting
    print(f"\nBefore lifting - Constraint active: {robot.grasp_constraint is not None}")
    if robot.grasp_constraint:
        print(f"   Constraint ID: {robot.grasp_constraint}")
    
    print("\n[Step 4] Lifting box to 50cm...")
    target_height = 0.5  # 50cm
    
    for i in range(1000):  # Enough steps to reach 50cm
        action = np.array([0, 0, 1.0, 0, 0, 0, 1.0])  # Max upward action
        robot.process_action(action)
        pyb.stepSimulation()
        ft_sensor.update(i)
        
        if i % 50 == 0:
            ee_state = pyb.getLinkState(robot_pyb_id, ee_link_index)
            current_pos = np.array(ee_state[0])
            
            # Get box position
            box_state = pyb.getBasePositionAndOrientation(box_pyb_id)
            box_pos = box_state[0]
            
            joint_force = ft_sensor.get_joint_force
            contact_force = ft_sensor.get_contact_force
            
            print(f"   Step {i}:")
            print(f"      End-effector height: {current_pos[2]:.4f}m")
            print(f"      Box height: {box_pos[2]:.4f}m (target: {target_height:.2f}m)")
            print(f"      Constraint active: {robot.grasp_constraint is not None}")
            print(f"      Joint Force Z: {joint_force[2]:.2f} N")
            print(f"      Contact Force: [{contact_force[0]:.2f}, {contact_force[1]:.2f}, {contact_force[2]:.2f}] N")
        
        time.sleep(0.01)
        
        # Check if target reached
        box_state = pyb.getBasePositionAndOrientation(box_pyb_id)
        if box_state[0][2] >= target_height:
            print(f"\n   ✅ Reached target height! Box at {box_state[0][2]:.3f}m")
            break
            print(f"      Box height: {box_pos[2]:.4f}m")
            print(f"      Constraint active: {robot.grasp_constraint is not None}")
            print(f"      Joint Force Z: {joint_force[2]:.2f} N (gravity load)")
            print(f"      Contact Force: [{contact_force[0]:.2f}, {contact_force[1]:.2f}, {contact_force[2]:.2f}] N")
        
        time.sleep(0.02)
    
    # Summary
    print("\n" + "=" * 60)
    print("Force/Torque Sensor Summary")
    print("=" * 60)
    print(f"Sensor Observation Space:")
    for key, value in ft_sensor.get_observation_space_element().items():
        print(f"  {key}: {value}")
    
    print(f"\nCurrent Readings:")
    obs = ft_sensor.get_observation()
    for key, value in obs.items():
        print(f"  {key}: {value}")
    
    print(f"\nLogging Data:")
    log_data = ft_sensor.get_data_for_logging()
    for key, value in log_data.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
    print("Test complete! Press Ctrl+C to exit...")
    print("=" * 60)
    
    try:
        while True:
            pyb.stepSimulation()
            ft_sensor.update(0)
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\n👋 Exiting...")
    
    pyb.disconnect()


if __name__ == "__main__":
    main()
