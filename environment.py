"""
Spring Slider Gymnasium Environment
====================================

A reinforcement learning environment for training agents to pull a spring-loaded
slider to target distances with optimal smoothness and accuracy.

Features:
- Continuous action space (force application)
- Variable stiffness levels
- Configurable target distances
- Reward based on accuracy, smoothness, and speed
- Compatible with Stable-Baselines3 and other RL libraries
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import os


class SpringSliderEnv(gym.Env):
    """
    Spring Slider Environment for motor learning research.

    Observation Space:
        - position: Current slider position [0, 0.30] m
        - velocity: Current slider velocity (unbounded)
        - target_distance: Target position in meters
        - stiffness: Current spring stiffness (normalized)
        - time_elapsed: Time since episode start (normalized)

    Action Space:
        - Continuous force to apply [-1, 1] (scaled internally)

    Episode ends when:
        - Block returns to rest position after reaching target area
        - Maximum time steps reached
        - Block gets stuck or goes out of bounds
    """

    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 60}

    def __init__(
            self,
            stiffness=300.0,
            target_distance=0.20,  # meters
            max_steps=500,
            render_mode=None,
            gui=False,
            target_tolerance=0.01,  # 1 cm tolerance for "success"
            reward_weights=None
    ):
        super().__init__()

        # Environment parameters
        self.stiffness = stiffness
        self.target_distance = target_distance
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.gui = gui
        self.target_tolerance = target_tolerance

        # Reward weights (accuracy, force smoothness, velocity smoothness, speed)
        if reward_weights is None:
            self.reward_weights = {
                'accuracy': 100.0,  # Penalty for distance error
                'peak_force': 0.01,  # Penalty for high peak force
                'peak_velocity': 1.0,  # Penalty for high peak velocity
                'time': 0.1,  # Penalty for taking too long
                'success_bonus': 50.0  # Bonus for being within tolerance
            }
        else:
            self.reward_weights = reward_weights

        # Physical constants
        self.max_force = 50.0  # Maximum force agent can apply (N)
        self.max_position = 0.30  # Maximum slider travel (m)
        self.reset_pos_threshold = 0.0005  # Consider "zero" position
        self.reset_vel_threshold = 0.001  # Consider "stopped"

        # Define action and observation spaces
        # Action: continuous force [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )

        # Observation: [position, velocity, target, stiffness_normalized, time_normalized]
        self.observation_space = spaces.Box(
            low=np.array([0.0, -5.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([0.30, 5.0, 0.30, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # Episode tracking
        self.current_step = 0
        self.max_distance_reached = 0.0
        self.peak_force = 0.0
        self.peak_velocity = 0.0
        self.has_reached_target_area = False
        self.time_to_target = None

        # PyBullet setup
        self.physics_client = None
        self.slider_id = None
        self.slider_joint_index = None
        self.target_line_id = None

        # Create URDF file
        self._create_urdf()

    def _create_urdf(self):
        """Create the URDF file for the slider mechanism."""
        urdf_content = """<?xml version="1.0"?>
<robot name="grooved_slider">

  <!-- TRACK BASE with GROOVE -->
  <link name="world">
    <!-- Main track surface -->
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="0.4 0.06 0.01"/>
      </geometry>
      <material name="gray">
        <color rgba="0.6 0.6 0.6 1"/>
      </material>
    </visual>
    <!-- Left rail (creates groove) -->
    <visual>
      <origin xyz="0.05 -0.018 0.01" rpy="0 0 0"/>
      <geometry>
        <box size="0.3 0.024 0.02"/>
      </geometry>
      <material name="dark_gray">
        <color rgba="0.4 0.4 0.4 1"/>
      </material>
    </visual>
    <!-- Right rail (creates groove) -->
    <visual>
      <origin xyz="0.05 0.018 0.01" rpy="0 0 0"/>
      <geometry>
        <box size="0.3 0.024 0.02"/>
      </geometry>
      <material name="dark_gray">
        <color rgba="0.4 0.4 0.4 1"/>
      </material>
    </visual>
    <!-- FILLED SECTION (no groove) -->
    <visual>
      <origin xyz="-0.147 0 0.01" rpy="0 0 0"/>
      <geometry>
        <box size="0.106 0.06 0.02"/>
      </geometry>
      <material name="filled_gray">
        <color rgba="0.5 0.5 0.5 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="0.4 0.06 0.01"/>
      </geometry>
    </collision>
    <collision>
      <origin xyz="-0.147 0 0.01" rpy="0 0 0"/>
      <geometry>
        <box size="0.106 0.06 0.02"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10.0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>

  <!-- RED BLOCK (fixed anchor) -->
  <link name="red_block">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="0.03 0.05 0.04"/>
      </geometry>
      <material name="red">
        <color rgba="0.9 0.2 0.2 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="0.03 0.05 0.04"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
  </link>

  <joint name="red_fixed" type="fixed">
    <parent link="world"/>
    <child link="red_block"/>
    <origin xyz="-0.18 0 0.025" rpy="0 0 0"/>
  </joint>

  <!-- BLUE MOVABLE BLOCK -->
  <link name="blue_block">
    <visual>
      <origin xyz="0 0 0.02" rpy="0 0 0"/>
      <geometry>
        <box size="0.04 0.04 0.04"/>
      </geometry>
      <material name="blue">
        <color rgba="0.2 0.5 0.9 1"/>
      </material>
    </visual>
    <visual>
      <origin xyz="0 0 -0.005" rpy="0 0 0"/>
      <geometry>
        <box size="0.035 0.014 0.015"/>
      </geometry>
      <material name="dark_blue">
        <color rgba="0.1 0.3 0.6 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0.02" rpy="0 0 0"/>
      <geometry>
        <box size="0.04 0.04 0.04"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.2"/>
      <origin xyz="0 0 0.02" rpy="0 0 0"/>
      <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
    </inertial>
  </link>

  <joint name="slider" type="prismatic">
    <parent link="world"/>
    <child link="blue_block"/>
    <origin xyz="-0.094 0 0.01" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="0" upper="0.30" effort="1000" velocity="5.0"/>
    <dynamics damping="0.05" friction="0.0"/>
  </joint>

</robot>
"""
        with open("grooved_slider.urdf", "w") as f:
            f.write(urdf_content)

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)

        # Handle options for curriculum learning
        if options is not None:
            if 'stiffness' in options:
                self.stiffness = options['stiffness']
            if 'target_distance' in options:
                self.target_distance = options['target_distance']

        # Disconnect previous session if exists
        if self.physics_client is not None:
            p.disconnect(self.physics_client)

        # Initialize PyBullet
        if self.gui or self.render_mode == 'human':
            self.physics_client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.resetDebugVisualizerCamera(
                cameraDistance=0.65,
                cameraYaw=90,
                cameraPitch=-20,
                cameraTargetPosition=[0, 0, 0.5]
            )
        else:
            self.physics_client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1.0 / 240.0)

        # Load environment
        plane_id = p.loadURDF("plane.urdf")
        self.slider_id = p.loadURDF(
            "grooved_slider.urdf",
            basePosition=[0, 0, 0.5],
            useFixedBase=True,
            flags=p.URDF_USE_SELF_COLLISION
        )

        # Enable self-collision
        num_joints = p.getNumJoints(self.slider_id)
        for i in range(-1, num_joints):
            for j in range(-1, num_joints):
                if i != j:
                    p.setCollisionFilterPair(self.slider_id, self.slider_id, i, j, 1)

        # Find slider joint
        self.slider_joint_index = None
        for i in range(num_joints):
            info = p.getJointInfo(self.slider_id, i)
            if b"slider" in info[1]:
                self.slider_joint_index = i
                break

        if self.slider_joint_index is None:
            raise RuntimeError("Could not find slider joint in URDF")

        # Reset joint to zero position
        p.resetJointState(self.slider_id, self.slider_joint_index, 0.0, 0.0)

        # Reset episode tracking
        self.current_step = 0
        self.max_distance_reached = 0.0
        self.peak_force = 0.0
        self.peak_velocity = 0.0
        self.has_reached_target_area = False
        self.time_to_target = None

        # Draw target line if in GUI mode
        if self.gui or self.render_mode == 'human':
            self._draw_target_line()

        # Return initial observation
        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action):
        """Execute one environment step."""
        # Scale action to force range
        force = float(action[0]) * self.max_force

        # Get current state before applying force
        joint_state = p.getJointState(self.slider_id, self.slider_joint_index)
        position = joint_state[0]
        velocity = joint_state[1]

        # Apply spring force + agent's force
        spring_force = -self.stiffness * position
        total_force = spring_force + force

        # Apply force to joint
        p.setJointMotorControl2(
            self.slider_id,
            self.slider_joint_index,
            p.VELOCITY_CONTROL,
            targetVelocity=0,
            force=0
        )
        p.setJointMotorControl2(
            self.slider_id,
            self.slider_joint_index,
            p.TORQUE_CONTROL,
            force=total_force
        )

        # Step simulation
        p.stepSimulation()

        # Update tracking metrics
        self.current_step += 1

        # Get new state
        joint_state = p.getJointState(self.slider_id, self.slider_joint_index)
        new_position = joint_state[0]
        new_velocity = joint_state[1]

        # Update max distance
        if new_position > self.max_distance_reached:
            self.max_distance_reached = new_position

        # Update peak force (absolute value of total force)
        current_force = abs(total_force)
        if current_force > self.peak_force:
            self.peak_force = current_force

        # Update peak velocity
        if abs(new_velocity) > self.peak_velocity:
            self.peak_velocity = abs(new_velocity)

        # Check if reached target area for the first time
        if not self.has_reached_target_area:
            if abs(new_position - self.target_distance) <= self.target_tolerance:
                self.has_reached_target_area = True
                self.time_to_target = self.current_step

        # Check termination conditions
        terminated = False
        truncated = False

        # Episode ends when block returns to rest after motion
        if self.max_distance_reached > 0.01:  # Has moved significantly
            if abs(new_position) < self.reset_pos_threshold and abs(new_velocity) < self.reset_vel_threshold:
                terminated = True

        # Truncate if max steps reached
        if self.current_step >= self.max_steps:
            truncated = True

        # Calculate reward
        reward = self._calculate_reward(new_position, new_velocity, terminated)

        # Get observation and info
        observation = self._get_observation()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _calculate_reward(self, position, velocity, episode_ended):
        """
        Calculate reward based on multiple objectives:
        - Accuracy: Distance from max_distance_reached to target
        - Smoothness: Peak force and peak velocity penalties
        - Speed: Time to complete the task
        """
        reward = 0.0

        # Only calculate final reward when episode ends
        if not episode_ended:
            # Small step reward for making progress toward target
            distance_to_target = abs(position - self.target_distance)
            step_reward = -0.01 * distance_to_target
            return step_reward

        # Accuracy component (main reward signal)
        distance_error = abs(self.max_distance_reached - self.target_distance)
        accuracy_penalty = -self.reward_weights['accuracy'] * distance_error
        reward += accuracy_penalty

        # Success bonus if within tolerance
        if distance_error <= self.target_tolerance:
            reward += self.reward_weights['success_bonus']

        # Smoothness penalties
        force_penalty = -self.reward_weights['peak_force'] * self.peak_force
        velocity_penalty = -self.reward_weights['peak_velocity'] * self.peak_velocity
        reward += force_penalty
        reward += velocity_penalty

        # Time penalty (encourage faster completion)
        time_penalty = -self.reward_weights['time'] * self.current_step
        reward += time_penalty

        return reward

    def _get_observation(self):
        """Get current observation."""
        joint_state = p.getJointState(self.slider_id, self.slider_joint_index)
        position = joint_state[0]
        velocity = joint_state[1]

        # Normalize stiffness to [0, 1] range (assuming max stiffness of 1000)
        normalized_stiffness = self.stiffness / 1000.0

        # Normalize time to [0, 1] range
        normalized_time = self.current_step / self.max_steps

        observation = np.array([
            position,
            velocity,
            self.target_distance,
            normalized_stiffness,
            normalized_time
        ], dtype=np.float32)

        return observation

    def _get_info(self):
        """Get additional information about the current state."""
        joint_state = p.getJointState(self.slider_id, self.slider_joint_index)
        position = joint_state[0]

        return {
            'current_position': position,
            'target_distance': self.target_distance,
            'stiffness': self.stiffness,
            'max_distance_reached': self.max_distance_reached,
            'peak_force': self.peak_force,
            'peak_velocity': self.peak_velocity,
            'distance_error': abs(self.max_distance_reached - self.target_distance),
            'has_reached_target': self.has_reached_target_area,
            'time_to_target': self.time_to_target,
            'current_step': self.current_step
        }

    def _draw_target_line(self):
        """Draw a green line showing the target distance."""
        if self.target_line_id is not None:
            try:
                p.removeUserDebugItem(self.target_line_id)
            except:
                pass

        # Calculate target position in world coordinates
        target_x = -0.094 + self.target_distance
        line_start = [target_x, -0.03, 0.525]
        line_end = [target_x, 0.03, 0.525]

        self.target_line_id = p.addUserDebugLine(
            line_start,
            line_end,
            lineColorRGB=[0, 1, 0],
            lineWidth=5,
            lifeTime=0
        )

    def render(self):
        """Render the environment."""
        if self.render_mode == 'rgb_array':
            # Get camera image
            view_matrix = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=[0, 0, 0.5],
                distance=0.65,
                yaw=90,
                pitch=-20,
                roll=0,
                upAxisIndex=2
            )
            proj_matrix = p.computeProjectionMatrixFOV(
                fov=60,
                aspect=1.0,
                nearVal=0.1,
                farVal=100.0
            )
            (_, _, px, _, _) = p.getCameraImage(
                width=640,
                height=480,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                renderer=p.ER_BULLET_HARDWARE_OPENGL
            )
            rgb_array = np.array(px, dtype=np.uint8)
            rgb_array = np.reshape(rgb_array, (480, 640, 4))
            rgb_array = rgb_array[:, :, :3]
            return rgb_array
        elif self.render_mode == 'human':
            # GUI is already rendering
            pass

    def close(self):
        """Clean up the environment."""
        if self.target_line_id is not None:
            try:
                p.removeUserDebugItem(self.target_line_id)
            except:
                pass

        if self.physics_client is not None:
            p.disconnect(self.physics_client)
            self.physics_client = None


# Register environment with Gymnasium
gym.register(
    id='SpringSlider-v0',
    entry_point='spring_slider_env:SpringSliderEnv',
    max_episode_steps=500,
)

if __name__ == "__main__":
    """Test the environment with random actions."""
    print("=" * 70)
    print("Testing Spring Slider Environment")
    print("=" * 70)

    # Create environment
    env = SpringSliderEnv(
        stiffness=300.0,
        target_distance=0.20,
        gui=True,
        render_mode='human'
    )

    print("\nEnvironment created successfully!")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")

    # Run a few episodes with random actions
    num_episodes = 3

    for episode in range(num_episodes):
        obs, info = env.reset()
        print(f"\n{'=' * 70}")
        print(f"Episode {episode + 1}")
        print(f"Target: {info['target_distance'] * 100:.1f} cm, Stiffness: {info['stiffness']:.0f} N/m")
        print(f"{'=' * 70}")

        episode_reward = 0
        done = False
        step = 0

        while not done:
            # Random action (you would replace this with your RL agent)
            action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            step += 1

            # Print progress every 50 steps
            if step % 50 == 0:
                print(f"Step {step}: Pos={info['current_position'] * 100:.2f}cm, "
                      f"MaxReached={info['max_distance_reached'] * 100:.2f}cm")

        # Episode summary
        print(f"\n{'=' * 70}")
        print(f"Episode {episode + 1} Complete!")
        print(f"Max distance reached: {info['max_distance_reached'] * 100:.2f} cm")
        print(f"Target distance: {info['target_distance'] * 100:.1f} cm")
        print(f"Error: {info['distance_error'] * 100:.2f} cm")
        print(f"Peak force: {info['peak_force']:.1f} N")
        print(f"Peak velocity: {info['peak_velocity']:.2f} m/s")
        print(f"Total steps: {info['current_step']}")
        print(f"Episode reward: {episode_reward:.2f}")

        if info['has_reached_target']:
            print(f"✓ Reached target area in {info['time_to_target']} steps")
        else:
            print("✗ Did not reach target area")
        print(f"{'=' * 70}")

    env.close()
    print("\n✓ Environment test complete!")