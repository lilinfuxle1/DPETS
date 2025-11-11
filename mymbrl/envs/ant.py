from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import random
import numpy as np
import torch

import mujoco
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from gymnasium.utils import seeding


class AntEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
            "rgbd_tuple",
        ],
    }

    MODEL_IN = 45   # 预处理后状态(35维) + 动作(8维)
    MODEL_OUT = 29  # 状态变化量Δs维度

    _healthy_z_range = (0.2, 1.0)  # 躯干z轴范围（低于0.2视为摔倒）
    _healthy_reward = 1.0          # 存活奖励
    _ctrl_cost_weight = 0.01       # 控制成本权重
    _contact_cost_weight = 5e-4    # 接触力成本权重
    _contact_force_range = (-1.0, 1.0)  # 接触力裁剪范围
    _reset_noise_scale = 0.1       # 重置噪声幅度
    _exclude_current_positions_from_observation = False  # 不排除躯干位置
    _include_cfrc_ext_in_observation = False  # 不包含接触力
    _terminate_when_unhealthy = True  # 摔倒终止
    _main_body = "torso"           # 躯干body名称
    _forward_reward_weight = 1.0   # 前向速度奖励权重

    def __init__(self, seed=None):
        utils.EzPickle.__init__(self)
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(27,),  # 原始观测维度
            dtype=np.float32
        )
        dir_path = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(dir_path, "assets", "ant.xml")

        super().__init__(
            model_path=model_path,
            frame_skip=5,
            observation_space=observation_space,
        )
        #self.seed(seed)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        random.seed(seed)
        np.random.seed(seed)
        return [seed]

    @property
    def healthy_reward(self):
        return self.is_healthy * self._healthy_reward

    def control_cost(self, action):
        return self._ctrl_cost_weight * np.sum(np.square(action))

    @property
    def contact_forces(self):
        raw_contact_forces = self.data.cfrc_ext
        min_value, max_value = self._contact_force_range
        return np.clip(raw_contact_forces, min_value, max_value)

    @property
    def contact_cost(self):
        return self._contact_cost_weight * np.sum(np.square(self.contact_forces))

    @property
    def is_healthy(self):
        state = self.state_vector()
        min_z, max_z = self._healthy_z_range
        return np.isfinite(state).all() and (min_z <= state[2] <= max_z)

    def step(self, action):
        action = self.clip_action(action)

        xy_position_before = self.data.body(self._main_body).xpos[:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.data.body(self._main_body).xpos[:2].copy()

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity = xy_velocity[0]

        obs = self._get_obs()
        reward, reward_info = self._get_rew(x_velocity, action)
        terminated = (not self.is_healthy) and self._terminate_when_unhealthy
        info = {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
            "x_velocity": x_velocity,
            **reward_info,
        }

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, False, info

    def _get_rew(self, x_velocity: float, action):
        forward_reward = x_velocity * self._forward_reward_weight
        healthy_reward = self.healthy_reward
        rewards = forward_reward + healthy_reward

        ctrl_cost = self.control_cost(action)
        contact_cost = self.contact_cost
        costs = ctrl_cost + contact_cost

        reward = rewards - costs

        reward_info = {
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_contact": -contact_cost,
            "reward_survive": healthy_reward,
        }
        return reward, reward_info

    def _get_obs(self):
        position = self.data.qpos.flatten()
        velocity = self.data.qvel.flatten()

        if self._exclude_current_positions_from_observation:
            position = position[2:]

        if self._include_cfrc_ext_in_observation:
            contact_force = self.contact_forces[1:].flatten()
            return np.concatenate((position, velocity, contact_force))
        else:
            return np.concatenate((position, velocity))

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.standard_normal(self.model.nv)
        self.set_state(qpos, qvel)

        return self._get_obs()

    def _get_reset_info(self):
        return {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
        }

    # 观测预处理，DPETS用，将角度变换为sin/cos，方便拟合周期性
    @staticmethod
    def obs_preproc(obs):
       
        obs = obs

        if isinstance(obs, np.ndarray):
            joint_angles = obs[..., 5:13]  # 8个关节角度
            joint_sin = np.sin(joint_angles)
            joint_cos = np.cos(joint_angles)
            non_periodic = np.concatenate([obs[..., :5], obs[..., 13:]], axis=-1)
            return np.concatenate([non_periodic, joint_sin, joint_cos], axis=-1)
        else:
            joint_angles = obs[..., 5:13]
            joint_sin = torch.sin(joint_angles)
            joint_cos = torch.cos(joint_angles)
            non_periodic = torch.cat([obs[..., :5], obs[..., 13:]], dim=-1)
            return torch.cat([non_periodic, joint_sin, joint_cos], dim=-1)

    # 观测后处理，预测增量加上当前obs得到下一步obs
    @staticmethod
    def obs_postproc(obs, pred):
        return obs + pred

    # 标签处理，目标是next_obs - obs，即状态变化量
    @staticmethod
    def targ_proc(obs, next_obs):
        return next_obs - obs

    # 观测相关的cost，DPETS中用于MPC成本设计
    @staticmethod
    def obs_cost_fn_cost(obs):
        if isinstance(obs, np.ndarray):
            pose_cost = np.sum(np.square(obs[..., 3:5]), axis=-1)  # 躯干角度限制
            speed_cost = -0.1 * obs[..., 13]  # 惩罚速度太低
            joint_cost = 0.05 * np.sum(np.square(np.clip(obs[..., 5:13], -np.pi/2, np.pi/2)), axis=-1)
            return pose_cost + speed_cost + joint_cost
        else:
            pose_cost = torch.sum(torch.square(obs[..., 3:5]), dim=-1)
            speed_cost = -0.1 * obs[..., 13]
            joint_cost = 0.05 * torch.sum(torch.square(torch.clip(obs[..., 5:13], -np.pi/2, np.pi/2)), dim=-1)
            return pose_cost + speed_cost + joint_cost

    # 动作cost函数，用于控制动作幅度的惩罚
    @staticmethod
    def ac_cost_fn_cost(acs):
        if isinstance(acs, np.ndarray):
            return 0.01 * np.sum(np.square(acs), axis=-1)
        else:
            return 0.01 * torch.sum(torch.square(acs), dim=-1)

    # 动作裁剪，保证动作在[-1,1]
    @staticmethod
    def clip_action(action):
        if isinstance(action, np.ndarray):
            return np.clip(action, -1.0, 1.0)
        else:
            return torch.clip(action, -1.0, 1.0)

    # 状态归一化，训练时使用
    @staticmethod
    def normalize_state(state):
        mean = np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0] + [0.0]*8 + [0.0]*3 + [0.0]*3 + [0.0]*8)
        std = np.array([5.0, 5.0, 0.1, 0.5, 0.5, 0.5] + [np.pi/2]*8 + [2.0]*3 + [1.0]*3 + [2.0]*8)
        if isinstance(state, np.ndarray):
            return (state - mean) / (std + 1e-8)
        else:
            mean_t = torch.tensor(mean, device=state.device)
            std_t = torch.tensor(std, device=state.device)
            return (state - mean_t) / (std_t + 1e-8)

    # 反归一化
    @staticmethod
    def denormalize_state(normalized_state):
        mean = np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0] + [0.0]*8 + [0.0]*3 + [0.0]*3 + [0.0]*8)
        std = np.array([5.0, 5.0, 0.1, 0.5, 0.5, 0.5] + [np.pi/2]*8 + [2.0]*3 + [1.0]*3 + [2.0]*8)
        if isinstance(normalized_state, np.ndarray):
            return normalized_state * std + mean
        else:
            mean_t = torch.tensor(mean, device=normalized_state.device)
            std_t = torch.tensor(std, device=normalized_state.device)
            return normalized_state * std_t + mean_t

    @staticmethod
    def viewer_setup(viewer):
        viewer.cam.trackbodyid = 1  # 跟踪躯干
        viewer.cam.distance = viewer.model.stat.extent * 2.0
        viewer.cam.elevation = -30