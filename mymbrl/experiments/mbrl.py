import mymbrl.agents as agents
import mymbrl.envs as envs
import numpy as np
import gymnasium as gym
from gymnasium.envs.mujoco import MujocoEnv
import torch
import os
from typing import Tuple, List
import numpy as np

class MBRL:
    def __init__(self, config, writer):
        Env = envs.get_item(config.env)
        # 初始化环境时可指定render_mode（如需要可视化），新版gymnasium推荐显式设置
        self.env = Env()  # 不传seed
        self.env.reset(seed=config.random_seed)  # 创建后调用reset传seed  # 如需渲染可加：render_mode="human"
        self.writer = writer
        if hasattr(self.env, '_terminate_when_unhealthy'):
            self.env._terminate_when_unhealthy = False
            print(f"已禁用 {config.env} 环境的early stop机制")
        Agent = agents.get_item(config.agent.name)
        self.agent = Agent(config, self.env, writer)
        self.config = config

    def run(self):
        random_ntrain_iters = self.config.experiment.random_ntrain_iters
        for i in range(random_ntrain_iters):
            rewards = self.one_exp(epoch=i-random_ntrain_iters+1, is_random=True)
            self.writer.add_scalar('mbrl/rewards', rewards.sum(), 0)
        
        for i in range(self.config.experiment.ntrain_iters):
            rewards = self.one_exp(i+1)
            print(f"epoch {i}, rewards sum: {rewards.sum()}")
            self.writer.add_scalar('mbrl/rewards', rewards.sum(), i+1)

    def one_exp(self, epoch=0, is_random=False):
        self.agent.set_epoch(epoch)
        self.agent.reset()
        if epoch > 0:
            self.agent.train()

        # 适配新版gymnasium：reset()返回(observation, info)，显式拆分
        cur_state, _ = self.env.reset()
        cur_state = np.atleast_1d(cur_state)  # 确保状态为1维数组
        actions = []
        rewards = []
        states = [cur_state]

        for step in range(self.config.experiment.horizon):
            self.agent.set_step(step)
            action = None

            if is_random:
                # 随机动作采样，确保为1维数组
                action = self.env.action_space.sample()
                action = np.atleast_1d(action)
            else:
                # 智能体动作生成，处理张量并转为1维数组
                action = self.agent.sample(cur_state)
                if hasattr(action, 'detach'):  # 处理PyTorch张量
                    action = action.detach().cpu().numpy()
                action = np.atleast_1d(action)

            # 适配新版gymnasium：step()返回(observation, reward, terminated, truncated, info)
            next_state, reward, terminated, truncated, info = self.env.step(action)
            next_state = np.atleast_1d(next_state)  # 确保下一状态为1维数组
            reward = np.atleast_1d(reward).item()  # 奖励转为标量

            # 记录单步奖励
            self.writer.add_scalar(f'mbrl/rewards/epoch{epoch}', reward, step)

            # 形状校验（防御性编程）
            assert action.ndim == 1, f"动作应为1维数组，实际形状：{action.shape}"
            assert next_state.ndim == 1, f"下一状态应为1维数组，实际形状：{next_state.shape}"
            assert isinstance(reward, (int, float)), f"奖励应为标量，实际类型：{type(reward)}"

            actions.append(action)
            states.append(next_state)
            rewards.append(reward)

            cur_state = next_state

            # 适配新版gymnasium：episode结束条件为terminated（任务完成）或truncated（超时）
            if terminated or truncated:
                # 可根据需要记录终止原因（可选）
                if terminated:
                    print(f"Epoch {epoch} 步骤 {step}：任务完成终止")
                if truncated:
                    print(f"Epoch {epoch} 步骤 {step}：超时截断")
                break  # 终止当前episode

        # 拼接数据为数组（确保形状统一）
        states = np.stack(states, axis=0)  # 形状：[步数+1, 状态维度]
        actions = np.stack(actions, axis=0)  # 形状：[步数, 动作维度]
        rewards = np.array(rewards)  # 形状：[步数,]

        # 数据形状最终校验
        assert states.ndim == 2, f"状态应为2维数组，实际形状：{states.shape}"
        assert actions.ndim == 2, f"动作应为2维数组，实际形状：{actions.shape}"
        assert rewards.ndim == 1, f"奖励应为1维数组，实际形状：{rewards.shape}"

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