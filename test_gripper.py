#!/usr/bin/env python3
"""
Simple script to test gripper open/close functionality.
Visualizes the gripper opening and closing near a target object.
"""
## to do
# grippper does not grasp the object, try to implement force/torque sensor

import sys
import time
import numpy as np
import pybullet as pyb

# Add project to path
sys.path.insert(0, '.')

from modular_drl_env.util.pybullet_util import pybullet_util as pyb_u
from modular_drl_env.robot.robot_implementations.ur5 import UR5_Gripper
from modular_drl_env.world.obstacles.shapes import Box

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
    print("Gripper Test Script")
    print("=" * 60)
    
    # Connect to PyBullet with GUI
    pyb_u.init(
        assets_path="./modular_drl_env/assets",
        display_mode=True,
        sim_step=0.01,
        gravity=[0, 0, -9.8]
    )
    
    # Set camera to get a good view of the gripper
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
    
    # Create a test object (box) to grasp - moved closer to robot
    test_box = Box(
        position=[0.3, 0.0, 0.8],  # Closer: Y=0.0 instead of 0.5, Z=0.8 (robot height)
        rotation=[0, 0, 0],
        trajectory=[],
        sim_step=0.01,
        sim_steps_per_env_step=1,
        velocity=0,
        halfExtents=[0.025, 0.025, 0.025],  # 5cm box (half extents)
        color=[1, 0, 0, 1],
        seen_by_obstacle_sensor=True
    )
    test_box.build()
    world.objects.append(test_box)
    
    print(f"✅ Created test box at position: {test_box.position}")
    
    # Create UR5 robot with gripper
    # Provide resting angles for all 8 joints (6 arm + 2 fingers)
    # Finger joints will start at 0 (closed) and be opened by the controller
    robot = UR5_Gripper(
        name="ur5_robot",
        id_num=0,
        world=world,
        sim_step=0.01,
        use_physics_sim=True,
        base_position=[0, 0, 0.63],
        base_orientation=[0, 0, 0, 1],  # Quaternion [x, y, z, w]
        resting_angles=np.array([0, -1.57, -1.57, -1.57, 1.57, 0, 0.015, 0.015]),  # Better starting pose: 8 joints
        control_mode=0,  # Explicitly use mode 0 for IK
        ik_xyz_delta=0.05,  # Increased from default 0.005 to 0.05 (10x larger steps)
        ik_rpy_delta=0.05,  # Increased from default 0.005
        gripper_control=True
    )
    robot.build()
    
    print(f"   Total controllable joints: {len(robot.all_joints_ids)}")
    print(f"   Arm joints being controlled: {len(robot.controlled_joints_ids)}")
    print(f"   Control mode: {robot.control_mode}")
    print(f"   Left finger joint index: {robot.left_finger_joint_index}")
    print(f"   Right finger joint index: {robot.right_finger_joint_index}")
    
    # Get the link index for end effector
    robot_pyb_id = pyb_u.to_pb(robot.object_id)
    ee_link_index = -1  # Default to last link
    for i in range(pyb.getNumJoints(robot_pyb_id)):
        link_info = pyb.getJointInfo(robot_pyb_id, i)
        link_name = link_info[12].decode('utf-8')
        if link_name == robot.end_effector_link_id:
            ee_link_index = i
            break
    
    # Create dynamic sensors for robot that update from PyBullet
    class DynamicPositionRotationSensor:
        def __init__(self, robot_id, link_index):
            self.robot_id = robot_id
            self.link_index = link_index
        
        @property
        def position(self):
            link_state = pyb.getLinkState(self.robot_id, self.link_index)
            return np.array(link_state[0])
        
        @property
        def rotation(self):
            link_state = pyb.getLinkState(self.robot_id, self.link_index)
            return np.array(link_state[1])  # Quaternion
    
    class DynamicJointsSensor:
        def __init__(self, robot_id, joint_indices):
            self.robot_id = robot_id
            self.joint_indices = joint_indices
        
        @property
        def joints_angles(self):
            angles = []
            for joint_idx in self.joint_indices:
                joint_state = pyb.getJointState(self.robot_id, joint_idx)
                angles.append(joint_state[0])
            return np.array(angles)
    
    robot.position_rotation_sensor = DynamicPositionRotationSensor(robot_pyb_id, ee_link_index)
    robot.joints_sensor = DynamicJointsSensor(robot_pyb_id, robot.controlled_joints_ids)
    robot.sensors = []
    
    print(f"✅ Created UR5_Gripper robot")
    print(f"   Gripper control enabled: {robot.gripper_control}")
    print(f"   Initial gripper state: {robot.gripper_state}")
    
    # Get the link index for end effector
    robot_pyb_id = pyb_u.to_pb(robot.object_id)
    ee_link_index = -1  # Default to last link
    
    # Debug: print all joints/links
    print(f"\n   Searching for end-effector link: '{robot.end_effector_link_id}'")
    print("   Available joints/links:")
    for i in range(pyb.getNumJoints(robot_pyb_id)):
        link_info = pyb.getJointInfo(robot_pyb_id, i)
        link_name = link_info[12].decode('utf-8')
        joint_name = link_info[1].decode('utf-8')
        print(f"      Joint {i}: {joint_name} -> Link: {link_name}")
        if link_name == robot.end_effector_link_id:
            ee_link_index = i
    
    print(f"   End-effector link index: {ee_link_index}")
    
    # Get box position
    box_pyb_id = pyb_u.to_pb(test_box.object_id)
    box_state = pyb.getBasePositionAndOrientation(box_pyb_id)
    box_pos = np.array(box_state[0])
    print(f"\n📦 Box position: {box_pos}")
    
    # Move robot end-effector to box (at the same height for grasping)
    target_pos = box_pos.copy()  # Same position as box center
    print(f"🎯 Moving end-effector to: {target_pos}")
    
    # Get current EE position
    ee_state = pyb.getLinkState(robot_pyb_id, ee_link_index)
    current_pos = np.array(ee_state[0])
    print(f"   Current EE position: {current_pos}")
    print(f"   Distance to target: {np.linalg.norm(target_pos - current_pos):.4f}m")
    
    # First, open the gripper
    print("\n[Step 1] Opening gripper...")
    for i in range(30):
        action = np.array([0, 0, 0, 0, 0, 0, -1.0])  # Open gripper
        robot.process_action(action)
        pyb.stepSimulation()
        time.sleep(0.02)
    
    print(f"   Gripper state after opening: {robot.gripper_state:.3f}")
    
    # Move towards box in small steps with larger deltas
    print("\n[Step 2] Moving towards box...")
    steps = 1000  # Increased to 1000 for enough time to reach target
    for i in range(steps):
        # Get current position
        ee_state = pyb.getLinkState(robot_pyb_id, ee_link_index)
        current_pos = np.array(ee_state[0])
        
        # Calculate delta to target
        delta = target_pos - current_pos
        distance = np.linalg.norm(delta)
        
        # If close enough (within grasp distance), stop moving
        if distance < 0.01:  # 1cm is close enough for grasping
            print(f"   ✅ Reached target at step {i}! Distance: {distance:.4f}m")
            break
        
        # Use constant-speed normalized movement until VERY close
        # This prevents the "slowing down" effect
        if distance > 0.02:  # Use constant speed until 2cm away
            delta_normalized = (delta / distance) * 0.5  # Move 0.5 units per step (constant speed)
        else:  # Only when VERY close (< 2cm), use smaller movements
            delta_normalized = (delta / distance) * 0.1  # Slower for final approach
        
        action = np.array([
            delta_normalized[0],
            delta_normalized[1],
            delta_normalized[2],
            0, 0, 0,  # No rotation change
            -1.0  # Keep gripper open
        ])
        
        robot.process_action(action)
        pyb.stepSimulation()
        time.sleep(0.01)  # Reduced from 0.02 to 0.01 for faster movement
        
        # Show progress more frequently
        if i % 50 == 0:
            print(f"   Step {i}: Distance to target = {distance:.4f}m, Position = {current_pos}")
            print(f"   Step {i}: Distance to target = {distance:.4f}m")
    
    # Final position check
    ee_state = pyb.getLinkState(robot_pyb_id, ee_link_index)
    current_pos = np.array(ee_state[0])
    print(f"✅ Reached position: {current_pos}")
    
    # Check distance to box
    distance_to_box = np.linalg.norm(current_pos - box_pos)
    print(f"   Distance from EE to box: {distance_to_box:.4f}m")
    
    # Test gripper closing to grasp the box
    print("\n" + "=" * 60)
    print("Grasping Box")
    print("=" * 60)
    
    # Close gripper to grasp the box
    print("\n[Step 3] Closing gripper to grasp box...")
    print("   WATCH THE GRIPPER - Fingers should move TOGETHER and grasp the box")
    
    # First, lock the arm in current position by getting current joint states
    # Get the actual integer indices for the 6 arm joints (0-5), excluding fingers (7-8)
    arm_joint_indices = [0, 1, 2, 3, 4, 5]  # UR5 arm joints
    current_joint_positions = []
    
    for joint_idx in arm_joint_indices:
        joint_state = pyb.getJointState(robot_pyb_id, joint_idx)
        current_joint_positions.append(joint_state[0])
    
    for i in range(60):  # More steps for full closure
        # Hold arm joints in place while closing gripper
        for j, joint_idx in enumerate(arm_joint_indices):
            pyb.setJointMotorControl2(
                robot_pyb_id,
                joint_idx,
                pyb.POSITION_CONTROL,
                targetPosition=current_joint_positions[j],
                force=150,  # Standard force for UR5
                maxVelocity=1.0  # Standard velocity
            )
        
        # Set gripper state to closed and update fingers
        robot.gripper_state = 1.0  # 1.0 = fully closed
        robot._control_finger_joints()  # Call without parameters
        
        pyb.stepSimulation()
        time.sleep(0.05)
        
        # Show finger movement every 15 steps
        if i % 15 == 0 and robot.left_finger_joint_index is not None:
            left_finger_state = pyb.getJointState(robot_pyb_id, robot.left_finger_joint_index)
            right_finger_state = pyb.getJointState(robot_pyb_id, robot.right_finger_joint_index)
            print(f"   Step {i}: Left={left_finger_state[0]:.4f}rad, Right={right_finger_state[0]:.4f}rad")
    
    # Check final state
    if robot.left_finger_joint_index is not None:
        left_finger_state = pyb.getJointState(robot_pyb_id, robot.left_finger_joint_index)
        right_finger_state = pyb.getJointState(robot_pyb_id, robot.right_finger_joint_index)
        print(f"   ✅ Left finger angle: {left_finger_state[0]:.4f}rad (0.0 = fully closed)")
        print(f"   ✅ Right finger angle: {right_finger_state[0]:.4f}rad (0.0 = fully closed)")
    
    print(f"   Gripper state: {robot.gripper_state:.3f} (1.0 = closed)")
    print(f"   Grasp constraint: {robot.grasp_constraint}")
    
    if robot.grasp_constraint is not None:
        print("   🎉 Successfully grasped the box!")
    else:
        print("   ⚠️  Did not grasp box (may need to be closer)")
    
    input("\n   Press ENTER to lift the box...")
    
    # Test 3: Lift the box
    print("\n[Step 4] Lifting box...")
    
    # Get current position after grasping
    ee_state = pyb.getLinkState(robot_pyb_id, ee_link_index)
    start_pos = np.array(ee_state[0])
    lift_target = start_pos + np.array([0, 0, 0.2])  # Lift 20cm straight up
    
    print(f"   Starting lift from: {start_pos}")
    print(f"   Target position: {lift_target}")
    
    for i in range(200):  # More steps for smooth lifting
        ee_state = pyb.getLinkState(robot_pyb_id, ee_link_index)
        current_pos = np.array(ee_state[0])
        delta = lift_target - current_pos
        distance = np.linalg.norm(delta)
        
        # Stop if reached target
        if distance < 0.02:
            print(f"   ✅ Reached lift target at step {i}!")
            break
        
        # Simple constant upward movement
        action = np.array([
            0,     # No X movement
            0,     # No Y movement
            0.05,  # Constant upward movement (5cm per step)
            0, 0, 0,
            1.0  # Keep gripper closed
        ])
        robot.process_action(action)
        pyb.stepSimulation()
        time.sleep(0.02)
        
        if i % 50 == 0:
            print(f"   Step {i}: Height = {current_pos[2]:.4f}m, Distance to target = {distance:.4f}m")
    
    print(f"✅ Lifted to: {current_pos}")
    
    # Check if box was lifted with gripper
    box_state_after = pyb.getBasePositionAndOrientation(box_pyb_id)
    box_pos_after = np.array(box_state_after[0])
    print(f"   Box position after lift: {box_pos_after}")
    
    if box_pos_after[2] > box_pos[2] + 0.05:
        print("   🎉 Box was successfully grasped and lifted!")
    else:
        print("   ⚠️  Box did not lift (gripper may not have grasped it)")
    
    # Keep window open
    print("\n" + "=" * 60)
    print("Test complete! Press Ctrl+C to exit...")
    print("=" * 60)
    
    try:
        while True:
            pyb.stepSimulation()
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\n👋 Exiting...")
    
    pyb.disconnect()


if __name__ == "__main__":
    main()
