from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
from gymnasium.spaces import Box  
import numpy as np
from gymnasium.envs.mujoco import MujocoEnv

from gymnasium import utils
import torch
import mujoco

class HopperEnv(MujocoEnv, utils.EzPickle):
    MODEL_IN, MODEL_OUT = 17, 11
    OBS_ADD_DIM = 0
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],

    }

    def __init__(self, seed=None):
        utils.EzPickle.__init__(self)
        observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(12,),  
            dtype=np.float32
        )
        self._reset_noise_scale = 0.01
        self._ctrl_cost_weight = 0.001
        self._forward_reward_weight = 1.0
        self._healthy_reward = 1.0
        self._terminate_when_unhealthy = True
        self._healthy_state_range = (-100.0, 100.0)
        self._healthy_z_range = (0.7, 2.0)
        self._healthy_angle_range = (-0.2, 0.2)
        self._exclude_current_positions_from_observation = True  # 排除躯干的 x 坐标

        dir_path = os.path.dirname(os.path.abspath(__file__))
        model_path = f"{dir_path}/assets/hopper.xml"

        super().__init__(
            model_path=model_path,
            frame_skip=5,
            observation_space=observation_space,
        )
    @property
    def healthy_reward(self):
        return (
            float(self.is_healthy or self._terminate_when_unhealthy)
            * self._healthy_reward
        )

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    @property
    def is_healthy(self):
        z, angle = self.data.qpos[1:3]
        state = self.state_vector()[2:]

        min_state, max_state = self._healthy_state_range
        min_z, max_z = self._healthy_z_range
        min_angle, max_angle = self._healthy_angle_range

        healthy_state = np.all(np.logical_and(min_state < state, state < max_state))
        healthy_z = min_z < z < max_z
        healthy_angle = min_angle < angle < max_angle

        is_healthy = all((healthy_state, healthy_z, healthy_angle))

        return is_healthy

    @property
    def terminated(self):
        terminated = not self.is_healthy if self._terminate_when_unhealthy else False
        return terminated

    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = np.clip(self.data.qvel.flat.copy(), -10, 10)

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()
        return observation

    def step(self, action):
        x_position_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt

        ctrl_cost = self.control_cost(action)

        forward_reward = self._forward_reward_weight * x_velocity
        healthy_reward = self.healthy_reward

        rewards = forward_reward + healthy_reward
        costs = ctrl_cost

        observation = self._get_obs()
        reward = rewards - costs
        terminated = self.terminated
        info = {
            "x_position": x_position_after,
            "x_velocity": x_velocity,
        }

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return observation, reward, terminated, False, info

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation
    @staticmethod
    def obs_preproc(obs):
        """
        对 Hopper 的观测进行预处理：
        - 将 3 个关节角度（hip, thigh, leg）展开为 sin/cos
        - 其余部分保持不变
        """
        if isinstance(obs, np.ndarray):
            joint_angles = obs[..., 3:6]  # 3 个角度变量
            joint_sin = np.sin(joint_angles)
            joint_cos = np.cos(joint_angles)
            non_periodic = np.concatenate([obs[..., :3], obs[..., 6:]], axis=-1)
            return np.concatenate([non_periodic, joint_sin, joint_cos], axis=-1)
        else:
            joint_angles = obs[..., 3:6]
            joint_sin = torch.sin(joint_angles)
            joint_cos = torch.cos(joint_angles)
            non_periodic = torch.cat([obs[..., :3], obs[..., 6:]], dim=-1)
            return torch.cat([non_periodic, joint_sin, joint_cos], dim=-1)

    @staticmethod
    def obs_postproc(obs, pred):

        return obs + pred

    @staticmethod
    def targ_proc(obs, next_obs):

        return next_obs - obs

    @staticmethod
    def obs_cost_fn_cost(obs):
        if isinstance(obs, np.ndarray):
            height_cost = -obs[..., 1]
            angle_cost = np.square(obs[..., 2])
            return height_cost + angle_cost
        else:
            height_cost = -obs[..., 1]
            angle_cost = torch.square(obs[..., 2])
            return height_cost + angle_cost

    @staticmethod
    def ac_cost_fn_cost(acs):

        if isinstance(acs, np.ndarray):
            return 0.01 * np.sum(np.square(acs), axis=-1)
        else:
            return 0.01 * torch.sum(torch.square(acs), dim=-1)

    @staticmethod
    def clip_action(action):
        if isinstance(action, np.ndarray):
            return np.clip(action, -1.0, 1.0)
        else:
            return torch.clip(action, -1.0, 1.0)
