#!/usr/bin/env python
# coding: utf-8

# # 時序差分(Temporal Difference)之SARSA

# In[1]:


# 載入相關套件
import itertools
import sys
from collections import defaultdict
from typing import Callable, DefaultDict, Tuple

import matplotlib
import matplotlib.style
import numpy as np
from lib import plotting
from lib.envs.windy_gridworld import (
    WindyGridworldEnv,
)

matplotlib.style.use('ggplot')  # 設定繪圖的風格

# In[2]:


# 建立環境
env = WindyGridworldEnv()

# In[4]:


# 試玩
print(env.reset())  # 重置
env.render()  # 更新畫面

print(env.step(1))  # 走下一步
env.render()  # 更新畫面

print(env.step(1))  # 走下一步
env.render()  # 更新畫面

print(env.step(1))  # 走下一步
env.render()  # 更新畫面

print(env.step(1))  # 走下一步
env.render()  # 更新畫面

print(env.step(1))  # 走下一步
env.render()  # 更新畫面

print(env.step(1))  # 走下一步
env.render()  # 更新畫面

# In[5]:


# 定義 ε-greedy策略
def make_epsilon_greedy_policy(Q: DefaultDict[int, np.ndarray], epsilon: float, nA: int) -> Callable[[int], np.ndarray]:
    def policy_fn(observation: int) -> np.ndarray:
        # 每個行動的機率初始化，均為 ε / n
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[observation])
        # 最佳行動的機率再加 1 - ε
        A[best_action] += 1.0 - epsilon
        return A

    return policy_fn


# In[15]:


# 定義 SARSA 策略
def sarsa(
    env: WindyGridworldEnv,
    num_episodes: int,
    discount_factor: float = 1.0,
    alpha: float = 0.5,
    epsilon: float = 0.1,
) -> Tuple[DefaultDict[int, np.ndarray], plotting.EpisodeStats]:
    # 行動值函數初始化
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    # 記錄 所有回合的長度及獎勵
    stats = plotting.EpisodeStats(episode_lengths=np.zeros(num_episodes), episode_rewards=np.zeros(num_episodes))

    # 使用 ε-greedy策略
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

    # 實驗 N 回合
    for i_episode in range(num_episodes):
        # 每 100 回合顯示除錯訊息
        if (i_episode + 1) % 100 == 0:
            print(f"\r {(i_episode + 1)}/{num_episodes}回合.", end="")
            sys.stdout.flush()  # 清除畫面

        # 開始依策略實驗
        state = env.reset()
        action_probs = policy(state)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

        # 每次走一步就更新狀態值
        for t in itertools.count():
            # 走一步
            next_state, reward, done, _ = env.step(action)

            # 選擇下一步行動
            next_action_probs = policy(next_state)
            next_action = np.random.choice(np.arange(len(next_action_probs)), p=next_action_probs)

            # 更新長度及獎勵
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t

            # 更新狀態值
            td_target = reward + discount_factor * Q[next_state][next_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta

            if done:
                break

            action = next_action
            state = next_state

    return Q, stats


# In[16]:


# 執行 SARSA 策略
Q, stats = sarsa(env, 200)

# In[18]:


# 顯示結果
fig = plotting.plot_episode_stats(stats)

# In[ ]:
