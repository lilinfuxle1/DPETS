import mymbrl.agents as agents
import mymbrl.envs as envs
import numpy as np
import gymnasium as gym
from gymnasium.envs.mujoco import MujocoEnv  # 直接导入Gymnasium的MujocoEnv
import torch
import os

class MBRL:
    def __init__(self, config, writer):
        Env = envs.get_item(config.env)
        env = Env(seed=config.random_seed)
        self.env = env
        self.writer = writer
        
        Agent = agents.get_item(config.agent.name)
        self.agent = Agent(config, env, writer)
        self.config = config
        # 新代码：只保留初始化时传seed，删掉后续调用
        self.env = Env(seed=config.random_seed)

    def run(self):
        random_ntrain_iters = self.config.experiment.random_ntrain_iters
        for i in range(random_ntrain_iters):
            rewards = self.one_exp(epoch=i-random_ntrain_iters+1, is_random = True)
            self.writer.add_scalar('mbrl/rewards', rewards.sum(), 0)
        
        for i in range(self.config.experiment.ntrain_iters):
            rewards = self.one_exp(i+1)
            print("epoch", i, "rewards", rewards.sum())
            self.writer.add_scalar('mbrl/rewards', rewards.sum(), i+1)
    '''''
    def one_exp(self, epoch = 0, is_random = False):

        self.agent.set_epoch(epoch)
        self.agent.reset()
        if epoch > 0:
            self.agent.train()

        cur_states = self.env.reset()
        actions = []
        rewards = []
        states = [cur_states]
        for step in range(self.config.experiment.horizon):
            self.agent.set_step(step)
            action = None
            if is_random:
                action = self.env.action_space.sample()
            else:
                action = self.agent.sample(cur_states)
            next_state, reward, done, info= self.env.step(action)
            
            self.writer.add_scalar('mbrl/rewards/epoch'+str(epoch), reward, step)
            
            actions.append(action)
            states.append(next_state)
            rewards.append(reward)
            
            cur_states = next_state

        states, actions, rewards = tuple(map(lambda l: np.stack(l, axis=0),
                                            (states, actions, rewards)))
        
        self.agent.add_data(states, actions)
        return rewards    
    '''''
    def one_exp(self, epoch=0, is_random=False):
        self.agent.set_epoch(epoch)
        self.agent.reset()
        if epoch > 0:
            self.agent.train()

        # 1. 修复：兼容新版 Gym/Gymnasium 的 reset 返回格式 (obs, info)，只取观测值
        cur_state, _ = self.env.reset()  # 关键：拆分 (obs, info)，忽略 info
        # 确保初始状态是 1 维数组（避免嵌套结构）
        cur_state = np.atleast_1d(cur_state)  # 处理标量/列表→1维数组
        actions = []
        rewards = []
        states = [cur_state]  # 初始状态存入列表，确保形状统一

        for step in range(self.config.experiment.horizon):
            self.agent.set_step(step)
            action = None

            if is_random:
                # 2. 修复：随机动作统一为 1 维数组（避免标量导致形状不匹配）
                action = self.env.action_space.sample()
                action = np.atleast_1d(action)  # 关键：确保 action 是 (n,) 形状（如 (1,)）
            else:
                # 3. 修复：智能体输出动作确保是 1 维数组
                action = self.agent.sample(cur_state)
                # 处理可能的张量/列表→1维数组，并兼容 CPU/GPU 张量
                if hasattr(action, 'detach'):  # 若为 PyTorch 张量
                    action = action.detach().cpu().numpy()
                action = np.atleast_1d(action)  # 统一为 (n,) 形状

            # 4. 与环境交互，确保 next_state 和 reward 形状统一
            next_state, reward, done, info = self.env.step(action)
            # 确保 next_state 是 1 维数组
            next_state = np.atleast_1d(next_state)
            # 确保 reward 是标量（避免数组导致拼接错误）
            reward = np.atleast_1d(reward).item()  # 关键：数组→标量

            # 记录日志（保持不变）
            self.writer.add_scalar('mbrl/rewards/epoch' + str(epoch), reward, step)

            # 存入列表前再次确认形状（防御性编程）
            assert action.ndim == 1, f"Action 应为1维，实际形状：{action.shape}"
            assert next_state.ndim == 1, f"Next state 应为1维，实际形状：{next_state.shape}"
            assert isinstance(reward, (int, float)), f"Reward 应为标量，实际类型：{type(reward)}"

            actions.append(action)
            states.append(next_state)
            rewards.append(reward)

            cur_state = next_state  # 更新当前状态，进入下一步

            # 5. 修复：若 episode 提前结束（done=True），终止循环（避免空步骤导致形状问题）
            if done:
                break

        # 6. 拼接数组：此时所有元素形状统一，可安全 stack
        # states 形状：[步数+1, 状态维度]（如 [201, 4]，包含初始状态）
        # actions 形状：[步数, 动作维度]（如 [200, 1]）
        # rewards 形状：[步数,]（如 [200,]，标量拼接为1维数组）
        states = np.stack(states, axis=0)
        actions = np.stack(actions, axis=0)
        rewards = np.array(rewards)  # 标量列表用 np.array 更高效（无需 stack）

        # 传入智能体的数据形状验证（可选，便于调试）
        assert states.ndim == 2, f"States 应为2维数组，实际形状：{states.shape}"
        assert actions.ndim == 2, f"Actions 应为2维数组，实际形状：{actions.shape}"
        assert rewards.ndim == 1, f"Rewards 应为1维数组，实际形状：{rewards.shape}"

        self.agent.add_data(states, actions)
        return rewards
