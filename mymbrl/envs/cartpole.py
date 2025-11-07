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


class CartpoleEnv(MujocoEnv, utils.EzPickle):
    PENDULUM_LENGTH = 0.6
    MODEL_IN, MODEL_OUT = 6, 4
    OBS_ADD_DIM = 1
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 25
    }
    

    def __init__(self, seed=None):
        
        utils.EzPickle.__init__(self)
        observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(4,),  
            dtype=np.float32
        )

        dir_path = os.path.dirname(os.path.abspath(__file__))
        model_path = f"{dir_path}/assets/cartpole.xml"

        super().__init__(
            model_path=model_path,
            frame_skip=2,
            observation_space=observation_space,
        )
        
        # if seed is not None:
        #     np.random.seed(seed)
        #     random.seed(seed)  # 新增：设置Python原生随机种子


    def _step(self, a):
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()

        cost_lscale = CartpoleEnv.PENDULUM_LENGTH
        reward = np.exp(
            -np.sum(np.square(self._get_ee_pos(ob) - np.array([0.0, CartpoleEnv.PENDULUM_LENGTH]))) / (cost_lscale **2)
        )
        reward -= 0.01 * np.sum(np.square(a))

        done = False
        info = {'reward': reward}  # 定义info字典，存储额外信息
        # _step只返回基础四要素（兼容内部逻辑）
        return ob.astype(np.float32), reward.astype(np.float32), done, info

    def step(self, action):
        # 调用_step获取基础数据
        ob, reward, done, info = self._step(action)
        # 适配新版Gymnasium：拆分done为terminated（任务完成）和truncated（超时截断）
        terminated = done  # cartpole无特殊终止条件，沿用done逻辑
        truncated = False  # 未设置超时，默认不截断
        # 返回新版要求的五要素
        return ob, reward, terminated, truncated, info

    def reset_model(self):
        qpos = self.init_qpos + np.random.normal(0, 0.1, np.shape(self.init_qpos))
        qvel = self.init_qvel + np.random.normal(0, 0.1, np.shape(self.init_qvel))
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel]).ravel()
    
    
    @staticmethod
    def _get_ee_pos(x):
        x0, theta = x[0], x[1]
        return np.array([
            x0 - CartpoleEnv.PENDULUM_LENGTH * np.sin(theta),
            -CartpoleEnv.PENDULUM_LENGTH * np.cos(theta)
        ])

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = v.model.stat.extent
        
    @staticmethod
    def obs_cost_fn_cost(obs):
        if isinstance(obs, np.ndarray):
            return -np.exp(-np.sum(
                np.square(CartpoleEnv._get_ee_pos_cost(obs) - np.array([0.0, 0.6])), axis=1
            ) / (0.6 ** 2))
        else:
            return -torch.exp(-torch.sum(
                torch.square(CartpoleEnv._get_ee_pos_cost(obs) - torch.tensor([0.0, 0.6], device=obs.device)), dim=1
            ) / (0.6 ** 2))

    @staticmethod
    def ac_cost_fn_cost(acs):
        if isinstance(acs, np.ndarray):
            return 0.01 * np.sum(np.square(acs), axis=1)
        else:
            return 0.01 * torch.sum(torch.square(acs), dim=1)
    
        
    @staticmethod
    def _get_ee_pos_cost(obs):
        x0, theta = obs[:, :1], obs[:, 1:2]
        if isinstance(obs, np.ndarray):
            return np.concatenate([x0 - 0.6 * np.sin(theta), -0.6 * np.cos(theta)], axis=1)
        else:
            return torch.cat([x0 - 0.6 * torch.sin(theta), -0.6 * torch.cos(theta)], dim=1)

    @staticmethod
    def obs_preproc(obs):
        if isinstance(obs, np.ndarray):
            dim = obs.ndim
            if dim == 1:
                return np.concatenate([np.sin(obs[1:2]), np.cos(obs[1:2]), obs[:1], obs[2:]], axis=-1)
            return np.concatenate([np.sin(obs[:, 1:2]), np.cos(obs[:, 1:2]), obs[:, :1], obs[:, 2:]], axis=1)
        else:
            dim = obs.ndim
            if dim == 3:
                return torch.cat((torch.sin(obs[:,:, 1:2]), torch.cos(obs[:,:, 1:2]), obs[:,:, :1], obs[:,:, 2:]), dim=2)
            return torch.cat((torch.sin(obs[:, 1:2]), torch.cos(obs[:, 1:2]), obs[:, :1], obs[:, 2:]), dim=1)

    @staticmethod
    def obs_postproc(obs, pred):
        return obs + pred

    @staticmethod
    def targ_proc(obs, next_obs):
        return next_obs - obs
