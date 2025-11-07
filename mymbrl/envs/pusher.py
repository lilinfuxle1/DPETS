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


class PusherEnv(MujocoEnv, utils.EzPickle):
    MODEL_IN, MODEL_OUT = 30, 23
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 20,
    }

    def __init__(self, seed=None):
        utils.EzPickle.__init__(self)
        
      
        observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(23,),  
            dtype=np.float32
        )
        
        
        dir_path = os.path.dirname(os.path.abspath(__file__))
        model_path = f"{dir_path}/assets/pusher.xml"  
        
      
        MujocoEnv.__init__(
            self,
            model_path=model_path,
            frame_skip=5,  
            observation_space=observation_space  # 仅保留支持的参数
        )
        # #种子设置
        # if seed is not None:
        #     np.random.seed(seed)
        #     random.seed(seed)          



    # def _step(self, a):
    #     obj_pos = self.get_body_com("object"),
    #     vec_1 = obj_pos - self.get_body_com("tips_arm")
    #     vec_2 = obj_pos - self.get_body_com("goal")

    #     reward_near = -np.sum(np.abs(vec_1))
    #     reward_dist = -np.sum(np.abs(vec_2))
    #     reward_ctrl = -np.square(a).sum()
    #     reward = 1.25 * reward_dist + 0.1 * reward_ctrl + 0.5 * reward_near

    #     self.do_simulation(a, self.frame_skip)
    #     ob = self._get_obs()
    #     done = False
    #     return ob, reward, done, {'reward': reward}

    def step(self, a):
        vec_1 = self.get_body_com("object") - self.get_body_com("tips_arm")
        vec_2 = self.get_body_com("object") - self.get_body_com("goal")

        reward_near = -np.linalg.norm(vec_1)
        reward_dist = -np.linalg.norm(vec_2)
        reward_ctrl = -np.square(a).sum()
        reward = 1.25*reward_dist + 0.1 * reward_ctrl + 0.5 * reward_near

        self.do_simulation(a, self.frame_skip)
        if self.render_mode == "human":
                self.render()

        ob = self._get_obs()
        return (
            ob,
            reward,
            False,
            False,
            dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl),
        )


    def viewer_setup(self):
        assert self.viewer is not None
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = 4.0

    def reset_model(self):
        qpos = self.init_qpos

       
        self.goal_pos = np.asarray([0, 0, 0.02])  #补z坐标
        while True:
            
            self.cylinder_pos = np.concatenate([
                self.np_random.uniform(low=-0.3, high=0, size=1),
                self.np_random.uniform(low=-0.2, high=0.2, size=1),
                np.array([0.02])  # 补z坐标，固定高度
            ])
            if np.linalg.norm(self.cylinder_pos[:2] - self.goal_pos[:2]) > 0.17:
                break

      
        qpos[-4:-1] = self.cylinder_pos  # 物体3维位置（原-4:-2→改为-4:-1，适配3维）
        qpos[-3:] = self.goal_pos        # 目标3维位置（原-2:→改为-3:，适配3维）
        qvel = self.init_qvel + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nv
        )
        qvel[-6:] = 0  
        self.set_state(qpos, qvel)
        
        
        self.ac_goal_pos = self.goal_pos  
        self.ac_goal_pos_tensor = torch.tensor(self.goal_pos, dtype=torch.float32)  # torch张量版本

        return self._get_obs()
    # def reset_model(self):
    #     qpos = self.init_qpos

    #     self.goal_pos = np.asarray([0, 0])
    #     self.cylinder_pos = np.array([-0.25, 0.15]) + np.random.normal(0, 0.025, [2])

    #     qpos[-4:-2] = self.cylinder_pos
    #     qpos[-2:] = self.goal_pos
    #     qvel = self.init_qvel + self.np_random.uniform(low=-0.005,
    #             high=0.005, size=self.model.nv)
    #     qvel[-4:] = 0
    #     self.set_state(qpos, qvel)
    #     self.ac_goal_pos = self.get_body_com("goal")
    #     # self.ac_goal_pos_tensor = torch.tensor(self.ac_goal_pos, device="cpu")

    #     return self._get_obs()

    def _get_obs(self):
        return np.concatenate([
            self.data.qpos.flat[:7],
            self.data.qvel.flat[:7],
            self.get_body_com("tips_arm"),
            self.get_body_com("object"),
            self.get_body_com("goal"),
        ])
        

        
    @staticmethod
    def obs_postproc(obs, pred):
        return obs + pred
    
    @staticmethod
    def obs_preproc(obs):
        return obs

    @staticmethod
    def targ_proc(obs, next_obs):
        return next_obs - obs

    def obs_cost_fn_cost(self, obs):
        ndim = obs.ndim
        to_w, og_w = 0.5, 1.25
        if ndim == 2:
            tip_pos, obj_pos, goal_pos = obs[:, 14:17], obs[:, 17:20], self.ac_goal_pos
        elif ndim == 3:
            tip_pos, obj_pos, goal_pos = obs[:, :, 14:17], obs[:, :, 17:20], self.ac_goal_pos

        if isinstance(obs, np.ndarray):
            tip_obj_dist = np.sum(np.abs(tip_pos - obj_pos), axis=-1)
            obj_goal_dist = np.sum(np.abs(goal_pos - obj_pos), axis=-1)
            return to_w * tip_obj_dist + og_w * obj_goal_dist
        else:
            # goal_pos = self.ac_goal_pos_tensor
            goal_pos = torch.tensor(goal_pos, device=obs.device)
            tip_obj_dist = torch.sum(torch.abs(tip_pos - obj_pos), dim=-1)
            obj_goal_dist = torch.sum(torch.abs(goal_pos - obj_pos), dim=-1)
            return to_w * tip_obj_dist + og_w * obj_goal_dist
    
    @staticmethod
    def ac_cost_fn_cost(acs):
        if isinstance(acs, np.ndarray):
            return 0.1 * np.sum(np.square(acs), axis=1)
        else:
            return 0.1 * torch.sum(torch.square(acs), dim=1)