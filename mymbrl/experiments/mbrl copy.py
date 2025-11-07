# import mymbrl.agents as agents
# import mymbrl.envs as envs
# import numpy as np
# import gymnasium as gym
# from gymnasium.envs.mujoco import MujocoEnv  # 直接导入Gymnasium的MujocoEnv
# import torch
# import os

# class MBRL:
#     def __init__(self, config, writer):
#         Env = envs.get_item(config.env)
#         env = Env(seed=config.random_seed)
#         self.env = env
#         self.writer = writer
        
#         Agent = agents.get_item(config.agent.name)
#         self.agent = Agent(config, env, writer)
#         self.config = config
#         # 新代码：只保留初始化时传seed，删掉后续调用
#         self.env = Env(seed=config.random_seed)

#     def run(self):
#         random_ntrain_iters = self.config.experiment.random_ntrain_iters
#         for i in range(random_ntrain_iters):
#             rewards = self.one_exp(epoch=i-random_ntrain_iters+1, is_random = True)
#             self.writer.add_scalar('mbrl/rewards', rewards.sum(), 0)
        
#         for i in range(self.config.experiment.ntrain_iters):
#             rewards = self.one_exp(i+1)
#             print("epoch", i, "rewards", rewards.sum())
#             self.writer.add_scalar('mbrl/rewards', rewards.sum(), i+1)
#     '''''
#     def one_exp(self, epoch = 0, is_random = False):

#         self.agent.set_epoch(epoch)
#         self.agent.reset()
#         if epoch > 0:
#             self.agent.train()

#         cur_states = self.env.reset()
#         actions = []
#         rewards = []
#         states = [cur_states]
#         for step in range(self.config.experiment.horizon):
#             self.agent.set_step(step)
#             action = None
#             if is_random:
#                 action = self.env.action_space.sample()
#             else:
#                 action = self.agent.sample(cur_states)
#             next_state, reward, done, info= self.env.step(action)
            
#             self.writer.add_scalar('mbrl/rewards/epoch'+str(epoch), reward, step)
            
#             actions.append(action)
#             states.append(next_state)
#             rewards.append(reward)
            
#             cur_states = next_state

#         states, actions, rewards = tuple(map(lambda l: np.stack(l, axis=0),
#                                             (states, actions, rewards)))
        
#         self.agent.add_data(states, actions)
#         return rewards    
#     '''''
#     def one_exp(self, epoch=0, is_random=False):
#         self.agent.set_epoch(epoch)
#         self.agent.reset()
#         if epoch > 0:
#             self.agent.train()

#         # 1. 修复：兼容新版 Gym/Gymnasium 的 reset 返回格式 (obs, info)，只取观测值
#         cur_state, _ = self.env.reset()  # 关键：拆分 (obs, info)，忽略 info
#         # 确保初始状态是 1 维数组（避免嵌套结构）
#         cur_state = np.atleast_1d(cur_state)  # 处理标量/列表→1维数组
#         actions = []
#         rewards = []
#         states = [cur_state]  # 初始状态存入列表，确保形状统一

#         for step in range(self.config.experiment.horizon):
#             self.agent.set_step(step)
#             action = None

#             if is_random:
#                 # 2. 修复：随机动作统一为 1 维数组（避免标量导致形状不匹配）
#                 action = self.env.action_space.sample()
#                 action = np.atleast_1d(action)  # 关键：确保 action 是 (n,) 形状（如 (1,)）
#             else:
#                 # 3. 修复：智能体输出动作确保是 1 维数组
#                 action = self.agent.sample(cur_state)
#                 # 处理可能的张量/列表→1维数组，并兼容 CPU/GPU 张量
#                 if hasattr(action, 'detach'):  # 若为 PyTorch 张量
#                     action = action.detach().cpu().numpy()
#                 action = np.atleast_1d(action)  # 统一为 (n,) 形状

#             # 4. 与环境交互，确保 next_state 和 reward 形状统一
#             next_state, reward, done, info = self.env.step(action)
#             # 确保 next_state 是 1 维数组
#             next_state = np.atleast_1d(next_state)
#             # 确保 reward 是标量（避免数组导致拼接错误）
#             reward = np.atleast_1d(reward).item()  # 关键：数组→标量

#             # 记录日志（保持不变）
#             self.writer.add_scalar('mbrl/rewards/epoch' + str(epoch), reward, step)

#             # 存入列表前再次确认形状（防御性编程）
#             assert action.ndim == 1, f"Action 应为1维，实际形状：{action.shape}"
#             assert next_state.ndim == 1, f"Next state 应为1维，实际形状：{next_state.shape}"
#             assert isinstance(reward, (int, float)), f"Reward 应为标量，实际类型：{type(reward)}"

#             actions.append(action)
#             states.append(next_state)
#             rewards.append(reward)

#             cur_state = next_state  # 更新当前状态，进入下一步

#             # 5. 修复：若 episode 提前结束（done=True），终止循环（避免空步骤导致形状问题）
#             if done:
#                 break

#         # 6. 拼接数组：此时所有元素形状统一，可安全 stack
#         # states 形状：[步数+1, 状态维度]（如 [201, 4]，包含初始状态）
#         # actions 形状：[步数, 动作维度]（如 [200, 1]）
#         # rewards 形状：[步数,]（如 [200,]，标量拼接为1维数组）
#         states = np.stack(states, axis=0)
#         actions = np.stack(actions, axis=0)
#         rewards = np.array(rewards)  # 标量列表用 np.array 更高效（无需 stack）

#         # 传入智能体的数据形状验证（可选，便于调试）
#         assert states.ndim == 2, f"States 应为2维数组，实际形状：{states.shape}"
#         assert actions.ndim == 2, f"Actions 应为2维数组，实际形状：{actions.shape}"
#         assert rewards.ndim == 1, f"Rewards 应为1维数组，实际形状：{rewards.shape}"

#         self.agent.add_data(states, actions)
#         return rewards
import mymbrl.agents as agents
import mymbrl.envs as envs
import numpy as np
import gymnasium as gym
import torch
import os
from typing import Tuple, List
from typing import List  # 导入List类型
import numpy as np

class MBRL:
    def __init__(self, config, writer):
        # 1. 初始化环境（Gymnasium 不支持 __init__ 传 seed，seed 在 reset 时传入）
        Env = envs.get_item(config.env)
        self.env = Env()  # 不传递 seed，避免参数错误
        self.writer = writer
        
        # 2. 初始化智能体（传入配置、环境、日志器）
        Agent = agents.get_item(config.agent.name)
        self.agent = Agent(config, self.env, writer)
        self.config = config
        self.random_seed = config.random_seed  # 保存种子，reset 时使用

    def run(self) -> np.ndarray:
        """启动完整实验流程：随机探索 → 正式训练"""
        # 阶段1：随机探索（收集初始数据）
        random_ntrain_iters = self.config.experiment.random_ntrain_iters
        for i in range(random_ntrain_iters):
            rewards = self._one_episode(
                epoch=i - random_ntrain_iters + 1,
                is_random=True
            )
            self.writer.add_scalar('mbrl/rewards/random_explore', rewards.sum(), i)
        
        # 阶段2：正式训练（交互+训练循环）
        exp_rewards = []
        for i in range(self.config.experiment.ntrain_iters):
            rewards = self._one_episode(epoch=i + 1, is_random=False)
            total_reward = rewards.sum()
            exp_rewards.append(total_reward)
            print(f"[Epoch {i+1}/{self.config.experiment.ntrain_iters}] Total Reward: {total_reward:.2f}")
            self.writer.add_scalar('mbrl/rewards/train', total_reward, i + 1)
        
        return np.array(exp_rewards)

    def _one_episode(self, epoch: int = 0, is_random: bool = False) -> np.ndarray:
        """执行单轮 Episode（从环境重置到结束），返回本轮奖励数组"""
        # 初始化智能体状态
        self.agent.set_epoch(epoch)
        self.agent.reset()
        if epoch > 0:
            self.agent.train()  # 仅训练阶段更新模型

        # 1. 环境重置（Gymnasium 格式：返回 (obs, info)，传入种子确保可复现）
        cur_state, _ = self.env.reset(seed=self.random_seed)
        # 统一状态格式：1维 float32 数组（适配模型输入）
        cur_state = self._standardize_array(cur_state, dtype=np.float32)
        assert cur_state.ndim == 1, f"初始状态必须为1维，实际维度：{cur_state.ndim}"

        # 初始化数据存储列表
        states: List[np.ndarray] = [cur_state]
        actions: List[np.ndarray] = []
        rewards: List[float] = []
        max_steps = self.config.experiment.horizon

        # 2. 单轮交互循环
        for step in range(max_steps):
            self.agent.set_step(step)
            
            # 生成动作（随机/智能体规划）
            action = self._generate_action(cur_state, is_random)
            
            # 3. 与环境交互（Gymnasium 格式：返回 5 个元素）
            next_state, reward, terminated, truncated, info = self.env.step(action)
            
            # 4. 数据格式标准化
            next_state = self._standardize_array(next_state, dtype=np.float32)
            reward = self._standardize_reward(reward)

            # 记录日志
            self.writer.add_scalar(f'mbrl/rewards/epoch_{epoch}', reward, step)

            # 存储数据
            actions.append(action)
            states.append(next_state)
            rewards.append(reward)

            # 更新当前状态
            cur_state = next_state

            # 5. 终止条件：Episode 结束（自然终止/步数用尽）
            if terminated or truncated:
                print(f"[Episode End] Step: {step+1} | Terminated: {terminated} | Truncated: {truncated}")
                break

        # 6. 数据格式转换（列表 → 数组，适配智能体数据接收格式）
        states = np.stack(states, axis=0)  # 形状：(步数+1, 状态维度)
        actions = np.stack(actions, axis=0)  # 形状：(步数, 动作维度)
        rewards = np.array(rewards, dtype=np.float32)  # 形状：(步数,)

        # 数据有效性校验（防御性编程）
        self._validate_data_shape(states, actions, rewards)

        # 7. 向智能体传入数据（用于训练动态模型）
        self.agent.add_data(states, actions)

        return rewards

    def _generate_action(self, cur_state: np.ndarray, is_random: bool) -> np.ndarray:
        """生成动作：随机动作（探索阶段）或智能体规划动作（训练阶段）"""
        if is_random:
            # 随机动作：从环境动作空间采样，标准化格式
            action = self.env.action_space.sample()
        else:
            # 智能体规划动作：处理张量→数组转换
            action = self.agent.sample(cur_state)
            if isinstance(action, torch.Tensor):
                action = action.detach().cpu().numpy()  # GPU→CPU→数组

        # 标准化动作格式：1维 float32 数组（适配环境输入）
        action = self._standardize_array(action, dtype=np.float32)
        assert action.ndim == 1, f"动作必须为1维，实际维度：{action.ndim}"
        return action

    @staticmethod
    def _standardize_array(arr: np.ndarray | List | float, dtype: np.dtype = np.float32) -> np.ndarray:
        """标准化数组格式：转为1维 numpy 数组，统一数据类型"""
        if not isinstance(arr, np.ndarray):
            arr = np.array(arr, dtype=dtype)
        return arr.squeeze().astype(dtype)  # 去除冗余维度，统一类型

    @staticmethod
    def _standardize_reward(reward: float | np.ndarray) -> float:
        """标准化奖励格式：转为标量 float"""
        if isinstance(reward, np.ndarray):
            reward = reward.item()  # 数组→标量
        return float(reward)

    @staticmethod
    def _validate_data_shape(states: np.ndarray, actions: np.ndarray, rewards: np.ndarray) -> None:
        """校验数据形状有效性"""
        assert states.ndim == 2, f"States 必须为2维（步数+1, 状态维度），实际维度：{states.ndim}"
        assert actions.ndim == 2, f"Actions 必须为2维（步数, 动作维度），实际维度：{actions.ndim}"
        assert rewards.ndim == 1, f"Rewards 必须为1维（步数,），实际维度：{rewards.ndim}"
        assert len(actions) == len(rewards), f"动作数与奖励数不匹配：动作数{len(actions)} vs 奖励数{len(rewards)}"
        assert len(states) == len(actions) + 1, f"状态数应为动作数+1：状态数{len(states)} vs 动作数{len(actions)}"