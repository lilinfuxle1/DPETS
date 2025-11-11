from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import random  # 新增：导入random模块，确保全量种子设置
import os
import mujoco  
import numpy as np
from gymnasium import utils  
from gymnasium.envs.mujoco import MujocoEnv  
from gymnasium.spaces import Box  
import torch

def mass_center(model, data):
    """Calculate center of mass as weighted average: (model.body_mass.T * data.xipos) / sum(model.body_mass)."""
    num = np.einsum("b,bj->j", model.body_mass, data.xipos)
    denom = model.body_mass.sum()
    return (num / denom)[0:2].copy()

class humanoidEnv(MujocoEnv, utils.EzPickle):
    MODEL_IN, MODEL_OUT = 393, 376
    OBS_ADD_DIM = 0
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
    }

    def __init__(self, seed=None):
        
        utils.EzPickle.__init__(self)
        self._forward_reward_weight = 1.25
        self._ctrl_cost_weight = 0.1
        self._healthy_reward = 5.0
        self._terminate_when_unhealthy = True
        self._healthy_z_range = (1.0, 2.0)

        self._reset_noise_scale = 1e-2

        self._exclude_current_positions_from_observation = True

        if self._exclude_current_positions_from_observation:
            observation_space = Box(
                low=-np.inf, high=np.inf, shape=(376,), dtype=np.float64
            )
        else:
            observation_space = Box(
                low=-np.inf, high=np.inf, shape=(378,), dtype=np.float64
            )

        dir_path = os.path.dirname(os.path.abspath(__file__))
        model_path = f"{dir_path}/assets/humanoid.xml"
        
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
        control_cost = self._ctrl_cost_weight * np.sum(np.square(self.data.ctrl))
        return control_cost

    @property
    def is_healthy(self):
        min_z, max_z = self._healthy_z_range
        is_healthy = min_z < self.data.qpos[2] < max_z

        return is_healthy

    @property
    def terminated(self):
        terminated = (not self.is_healthy) if self._terminate_when_unhealthy else False
        return terminated

    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()

        com_inertia = self.data.cinert.flat.copy()
        com_velocity = self.data.cvel.flat.copy()

        actuator_forces = self.data.qfrc_actuator.flat.copy()
        external_contact_forces = self.data.cfrc_ext.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[2:]

        return np.concatenate(
            (
                position,
                velocity,
                com_inertia,
                com_velocity,
                actuator_forces,
                external_contact_forces,
            )
        )

    def step(self, action):
        xy_position_before = mass_center(self.model, self.data)
        self.do_simulation(action, self.frame_skip)
        xy_position_after = mass_center(self.model, self.data)

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        ctrl_cost = self.control_cost(action)

        forward_reward = self._forward_reward_weight * x_velocity
        healthy_reward = self.healthy_reward

        rewards = forward_reward + healthy_reward

        observation = self._get_obs()
        reward = rewards - ctrl_cost
        terminated = self.terminated
        info = {
            "reward_linvel": forward_reward,
            "reward_quadctrl": -ctrl_cost,
            "reward_alive": healthy_reward,
            "x_position": xy_position_after[0],
            "y_position": xy_position_after[1],
            "distance_from_origin": np.linalg.norm(xy_position_after, ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            "forward_reward": forward_reward,
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
        return obs

    @staticmethod
    def obs_postproc(obs, pred):
        return obs + pred


    @staticmethod
    def targ_proc(obs, next_obs):
        return next_obs - obs
    
    @staticmethod
    def obs_cost_fn_cost(obs):
        """
        观测成本的简化版本：
        - 惩罚低高度：obs[..., 0] 对应 position[2:] 的第一项，即 torso 高度 z
        - 惩罚大姿态偏离：用前几维的平方和作为稳定项（避免过分依赖具体索引映射）
        说明：
        - Humanoid 的观测拼接顺序为 [position[2:], velocity, cinert, cvel, qfrc_actuator, cfrc_ext]
        - 这里仅使用高度与一个温和的 L2 姿态项，避免错误索引造成不稳定
        """
        if isinstance(obs, np.ndarray):
            height = obs[..., 0]
            posture_term = np.sum(np.square(obs[..., :10]), axis=-1)  # 温和姿态正则
            height_cost = -height
            return height_cost + 0.01 * posture_term
        else:
            height = obs[..., 0]
            posture_term = torch.sum(torch.square(obs[..., :10]), dim=-1)
            height_cost = -height
            return height_cost + 0.01 * posture_term

    # 动作成本：惩罚动作幅度
    @staticmethod
    def ac_cost_fn_cost(acs):
        if isinstance(acs, np.ndarray):
            return 0.01 * np.sum(np.square(acs), axis=-1)
        else:
            return 0.01 * torch.sum(torch.square(acs), dim=-1)

    # 动作裁剪：保证动作在 [-1, 1]
    @staticmethod
    def clip_action(action):
        if isinstance(action, np.ndarray):
            return np.clip(action, -1.0, 1.0)
        else:
            return torch.clip(action, -1.0, 1.0)



