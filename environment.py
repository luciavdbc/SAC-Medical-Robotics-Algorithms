"""
Medical Tissue Manipulation Environment
========================================

A reinforcement learning environment simulating surgical tissue manipulation
where the agent must handle tissues of varying stiffness.

Environment structure follows the Gymnasium API:
https://gymnasium.farama.org/

Physics based on Hooke's Law (F = -kx) for tissue elasticity.
PyBullet physics engine used for simulation: https://pybullet.org/

Compatible with Stable-Baselines3 for SAC training:
https://stable-baselines3.readthedocs.io/
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import os


class SpringSliderEnv(gym.Env):
    """
    Spring-loaded slider environment representing tissue manipulation.
    
    The agent controls forces applied to a block attached to a spring, representing tissue.
    Goal is to pull the block to a target distance then let it return to rest.
    Different spring stiffness values represent soft to stiff tissues.

    Observation Space:
        - position: Current slider position [0, 0.30] m
        - velocity: Current slider velocity
        - target_distance: Target position in meters
        - stiffness: Current spring stiffness (normalized)
        - time_elapsed: Time since episode start (normalized)

    Action Space:
        - Continuous force [-1, 1] scaled to actual force range

    Episode Termination:
        - Block returns to rest after reaching target
        - Maximum timesteps reached
    """

    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 60}

    def __init__(
            self,
            stiffness=300.0,
            target_distance=0.20,
            max_steps=500,
            render_mode=None,
            gui=False,
            target_tolerance=0.01,
            reward_weights=None
    ):
        super().__init__()

        # Environment parameters
        self.stiffness = stiffness
        self.target_distance = target_distance
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.gui = gui
        self.target_tolerance = target_tolerance  # 1 cm tolerance

        # Reward shaping weights
        if reward_weights is None:
            self.reward_weights = {
                'accuracy': 100.0,      
                'peak_force': 0.01,     
                'peak_velocity': 1.0,  
                'time': 0.1,           
                'success_bonus': 50.0   
            }
        else:
            self.reward_weights = reward_weights

        # Physical constants
        self.max_force = 50.0              
        self.max_position = 0.30          
        self.reset_pos_threshold = 0.0005  
        self.reset_vel_threshold = 0.001  

        # Action space: continuous force in [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )

        # Observation: [position, velocity, target, stiffness_norm, time_norm]
        self.observation_space = spaces.Box(
            low=np.array([0.0, -5.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([0.30, 5.0, 0.30, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        self.current_step = 0
        self.max_distance_reached = 0.0
        self.peak_force = 0.0
        self.peak_velocity = 0.0
        self.has_reached_target_area = False
        self.time_to_target = None

        self.physics_client = None
        self.slider_id = None
        self.slider_joint_index = None
        self.target_line_id = None

        # Create URDF file for PyBullet
        self._create_urdf()

    def _create_urdf(self):
        urdf_content = """<?xml version="1.0"?>
<robot name="grooved_slider">

  <!-- Track base with groove for slider -->
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
    <!-- Left rail -->
    <visual>
      <origin xyz="0.05 -0.018 0.01" rpy="0 0 0"/>
      <geometry>
        <box size="0.3 0.024 0.02"/>
      </geometry>
      <material name="dark_gray">
        <color rgba="0.4 0.4 0.4 1"/>
      </material>
    </visual>
    <!-- Right rail -->
    <visual>
      <origin xyz="0.05 0.018 0.01" rpy="0 0 0"/>
      <geometry>
        <box size="0.3 0.024 0.02"/>
      </geometry>
      <material name="dark_gray">
        <color rgba="0.4 0.4 0.4 1"/>
      </material>
    </visual>
    <!-- Filled section (no groove) -->
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

  <!-- Red block: fixed anchor point -->
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

  <!-- Blue block: movable slider -->
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

  <!-- Prismatic joint allows sliding motion -->
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
        super().reset(seed=seed)

        # Reset tracking variables
        self.current_step = 0
        self.max_distance_reached = 0.0
        self.peak_force = 0.0
        self.peak_velocity = 0.0
        self.has_reached_target_area = False
        self.time_to_target = None

        if self.physics_client is None:
            if self.gui:
                self.physics_client = p.connect(p.GUI)
                p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            else:
                self.physics_client = p.connect(p.DIRECT)

            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.setGravity(0, 0, -9.81)
            p.setTimeStep(1. / 240.)

            # Camera setup for visualization
            if self.gui:
                p.resetDebugVisualizerCamera(
                    cameraDistance=0.65,
                    cameraYaw=90,
                    cameraPitch=-20,
                    cameraTargetPosition=[0, 0, 0.5]
                )

            # Load environment
            planeId = p.loadURDF("plane.urdf")
            self.slider_id = p.loadURDF(
                "grooved_slider.urdf",
                basePosition=[0, 0, 0.5],
                useFixedBase=True,
                flags=p.URDF_USE_SELF_COLLISION
            )

            num_joints = p.getNumJoints(self.slider_id)
            for i in range(num_joints):
                joint_info = p.getJointInfo(self.slider_id, i)
                if b"slider" in joint_info[1]:
                    self.slider_joint_index = i
                    break

        # Reset slider to starting position
        p.resetJointState(self.slider_id, self.slider_joint_index, 0.0, 0.0)

        # Draw target line
        self._draw_target_line()

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action):
        """
        Execute one timestep of the environment.
        
        Applies force from action, simulates spring dynamics using Hooke's Law,
        and returns observation, reward, and termination status.
        """
        self.current_step += 1

        # Scale action to force range
        force = action[0] * self.max_force

        # Get current state
        joint_state = p.getJointState(self.slider_id, self.slider_joint_index)
        old_position = joint_state[0]
        old_velocity = joint_state[1]

        # Apply spring force (Hooke's Law: F = -kx)
        spring_force = -self.stiffness * old_position

        # Total force on slider
        total_force = force + spring_force

        # Track peak values for reward calculation
        self.peak_force = max(self.peak_force, abs(force))
        self.peak_velocity = max(self.peak_velocity, abs(old_velocity))

        # Applying force to joint
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

        p.stepSimulation()

        # Get new state
        joint_state = p.getJointState(self.slider_id, self.slider_joint_index)
        new_position = joint_state[0]
        new_velocity = joint_state[1]

        # Track maximum distance reached by block
        if new_position > self.max_distance_reached:
            self.max_distance_reached = new_position

            # Check if target area has been reached 
            if not self.has_reached_target_area:
                distance_to_target = abs(new_position - self.target_distance)
                if distance_to_target <= self.target_tolerance:
                    self.has_reached_target_area = True
                    self.time_to_target = self.current_step

        terminated = False
        truncated = False

        # Episode ends when block returns to rest after moving
        if self.max_distance_reached > 0.01:
            if abs(new_position) < self.reset_pos_threshold and abs(new_velocity) < self.reset_vel_threshold:
                terminated = True

        if self.current_step >= self.max_steps:
            truncated = True

        # Calculate reward
        reward = self._calculate_reward(new_position, new_velocity, terminated)

        observation = self._get_observation()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _calculate_reward(self, position, velocity, episode_ended):
        """
        Reward function based on accuracy, smoothness and speed.
        Main reward given at episode end based on how close we got to target.
        """
        reward = 0.0

        # During episode: small step reward for progress
        if not episode_ended:
            distance_to_target = abs(position - self.target_distance)
            step_reward = -0.01 * distance_to_target
            return step_reward

        # At episode end: calculate final reward based on performance
        distance_error = abs(self.max_distance_reached - self.target_distance)
        accuracy_penalty = -self.reward_weights['accuracy'] * distance_error
        reward += accuracy_penalty

        # Bonus for high accuracy
        if distance_error <= self.target_tolerance:
            reward += self.reward_weights['success_bonus']

        # Penalties for rough motion
        force_penalty = -self.reward_weights['peak_force'] * self.peak_force
        velocity_penalty = -self.reward_weights['peak_velocity'] * self.peak_velocity
        reward += force_penalty
        reward += velocity_penalty

        # Small penalty for taking longer
        time_penalty = -self.reward_weights['time'] * self.current_step
        reward += time_penalty

        return reward

    def _get_observation(self):
        """Get current state observation."""
        joint_state = p.getJointState(self.slider_id, self.slider_joint_index)
        position = joint_state[0]
        velocity = joint_state[1]

        # Normalize stiffness to [0, 1], assuming max 1000 N/m
        normalized_stiffness = self.stiffness / 1000.0

        # Normalize time
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
        """Draw visual marker showing target distance."""
        if self.target_line_id is not None:
            try:
                p.removeUserDebugItem(self.target_line_id)
            except:
                pass

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
        if self.render_mode == 'rgb_array':
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
            pass  

    def close(self):
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
    """Test environment with random actions."""
    print("=" * 70)
    print("Testing Spring Slider Environment")
    print("=" * 70)

    env = SpringSliderEnv(
        stiffness=300.0,
        target_distance=0.20,
        gui=True,
        render_mode='human'
    )

    print("\nEnvironment created")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")

    # Run test episodes
    num_episodes = 3

    for episode in range(num_episodes):
        obs, info = env.reset()
        print(f"\n{'=' * 70}")
        print(f"Episode {episode + 1}")
        print(f"Target: {info['target_distance'] * 100:.1f} cm")
        print(f"Stiffness: {info['stiffness']:.0f} N/m")
        print(f"{'=' * 70}")

        episode_reward = 0
        done = False
        step = 0

        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            step += 1

            if step % 50 == 0:
                print(f"Step {step}: Position={info['current_position'] * 100:.2f}cm, "
                      f"Max={info['max_distance_reached'] * 100:.2f}cm")

        # Episode summary
        print(f"\nEpisode Complete")
        print(f"Max distance: {info['max_distance_reached'] * 100:.2f} cm")
        print(f"Target: {info['target_distance'] * 100:.1f} cm")
        print(f"Error: {info['distance_error'] * 100:.2f} cm")
        print(f"Peak force: {info['peak_force']:.1f} N")
        print(f"Peak velocity: {info['peak_velocity']:.2f} m/s")
        print(f"Steps: {info['current_step']}")
        print(f"Reward: {episode_reward:.2f}")

        if info['has_reached_target']:
            print(f"Reached target in {info['time_to_target']} steps")

    env.close()
    print("\nTest complete")
