# 🎯 Simple Demo Environment - Quick Start Guide

## What We Created

A new custom environment called **SimpleDemoWorld** that demonstrates how to create your own robotic manipulation environments in this framework.

### Files Created:
1. **`modular_drl_env/world/world_implementations/simple_demo.py`** - The world implementation
2. **`configs/simple_demo_config.yaml`** - Configuration file for the environment

### What's in SimpleDemoWorld:
- ✅ Random boxes (customizable count)
- ✅ Random spheres (customizable count)
- ✅ Optional table
- ✅ Goal position marker (green sphere when visualizing)
- ✅ Workspace boundary visualization
- ✅ UR5 robot with LIDAR sensor
- ✅ Obstacle avoidance task

---

## 🚀 How to Run

### Evaluation Mode (with visualization):
```bash
conda activate drl
python3 run.py configs/simple_demo_config.yaml --eval
```

### Training Mode:
```bash
conda activate drl
python3 run.py configs/simple_demo_config.yaml --train
```

### Debug Mode (step-by-step):
```bash
conda activate drl
python3 run.py configs/simple_demo_config.yaml --debug
```

---

## ⚙️ Customization Options

Edit `configs/simple_demo_config.yaml` to customize:

### World Settings:
```yaml
world:
  type: "SimpleDemoWorld"
  config:
    workspace_boundaries: [-0.5, 0.5, -0.5, 0.5, 0.5, 1.0]  # [x_min, x_max, y_min, y_max, z_min, z_max]
    num_boxes: 3          # Number of box obstacles
    num_spheres: 2        # Number of sphere obstacles
    add_table: False      # Add a table to the scene
```

### Robot Settings:
```yaml
robots:
  - type: "UR5"          # Try: UR5, KR16, Kukaiiwa, KukaKr3, KR120
    config:
      base_position: [0, 0, 0.5]
      control_mode: 0    # 0=IK, 1=joint positions, 2=joint velocities
```

### Sensor Settings:
```yaml
sensors:
  - type: "LidarSensorUR5"  # Or try camera sensors
    config:
      ray_end: 0.4       # LIDAR range in meters
      indicator_buckets: 6
```

### Training Settings:
```yaml
train:
  num_envs: 4          # Parallel environments
  timesteps: 500000    # Total training steps
  save_freq: 10000     # Save every N steps
```

---

## 🎨 What You Can Render/Visualize

When running in **eval mode** with display enabled:

1. **Robot**: UR5 robotic arm
2. **Obstacles**: Random boxes and spheres
3. **Goal Marker**: Green semi-transparent sphere
4. **Workspace Boundaries**: White lines showing the valid workspace
5. **LIDAR Rays**: (if `show_sensor_aux: True`) Visual representation of distance sensors

---

## 📝 Creating Your Own Environment

Based on `simple_demo.py`, here's the structure:

```python
class YourWorld(World):
    def __init__(self, workspace_boundaries, ...):
        # Initialize your world parameters
        
    def set_up(self):
        # Build all objects ONCE (pre-generate)
        # Add ground, obstacles, etc.
        
    def reset(self, success_rate):
        # Reset for each episode
        # Move objects to new positions
        # Generate goal targets
        
    def update(self):
        # Update dynamic objects each step
        # For static worlds, just pass
        
    def build_visual_aux(self):
        # Optional: Add visual debugging aids
```

### Key Methods:
- **`set_up()`**: Called once at initialization - create all objects
- **`reset()`**: Called each episode - reposition objects and set goals
- **`update()`**: Called each step - update moving objects
- **`build_visual_aux()`**: Optional visual debugging

---

## 🎓 Next Steps

1. **Modify obstacle count**: Change `num_boxes` and `num_spheres` in config
2. **Try different robots**: Change `type: "UR5"` to `"KR16"` or others
3. **Add camera sensors**: Replace LIDAR with camera in config
4. **Train your agent**: Run with `--train` flag
5. **Create custom world**: Copy `simple_demo.py` and modify

---

## 🐛 Troubleshooting

**PyBullet window not showing?**
- Set `display: True` in the config (under `env:`)

**Robot not moving?**
- No model loaded in eval mode - it uses an untrained PPO by default
- Train first, then load the model for evaluation

**Collisions immediately?**
- Increase workspace boundaries
- Reduce number of obstacles
- Check `workspace_boundaries` in config

---

## 📚 Available Components

### Robots:
`UR5`, `UR5_Gripper`, `KR16`, `Kukaiiwa`, `KukaKr3`, `KR120`

### Sensors:
- **LIDAR**: `LidarSensorUR5`, `LidarSensorKR16`, `LidarSensorGeneric`
- **Cameras**: `OnBodyUR5`, `Floating`, `FloatingFollowEffector`
- **Positional**: `Joints`, `PositionRotation`, `RobotSkeleton`, `Obstacle`

### Goals:
`PositionCollision`, `PositionCollisionTrajectory`

### Worlds:
`SimpleDemoWorld`, `RandomObstacle`, `RandomBoxes`, `TableExperiment`, `KukaShelfExperiment`, and more!

---

Happy experimenting! 🚀
