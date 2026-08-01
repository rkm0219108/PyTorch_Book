#!/usr/bin/env python
# coding: utf-8

# In[8]:


# 載入相關套件
import gymnasium as gym
from gymnasium import envs

# In[9]:


# 已註冊的遊戲
all_envs = envs.registry.values()
env_ids = [env_spec.id for env_spec in all_envs]
print(env_ids)

# In[10]:


len(env_ids)

# In[15]:


import warnings

warnings.filterwarnings('ignore')

# In[16]:


# 列出環境資訊
import pandas as pd

space_names = ['觀測空間', '動作空間', '獎勵範圍', '最大步數']
df = pd.DataFrame(columns=space_names)

env_specs = gym.envs.registry.values()
no = 450  # len(env_ids)
i = 0
for env_spec in env_specs:
    i += 1
    if i > no:
        break
    env_id = env_spec.id
    try:
        env = gym.make(env_id)
        observation_space = env.observation_space
        action_space = env.action_space
        reward_range = env.reward_range
        max_episode_steps = None
        if isinstance(env, gym.wrappers.TimeLimit):
            max_episode_steps = env._max_episode_steps
        df.loc[env_id] = [observation_space, action_space, reward_range, max_episode_steps]
    except:
        pass

df.reset_index(level=0, inplace=True)
col = list(df.columns)
col[0] = 'name'
df.columns = col
with pd.option_context('display.max_rows', None):
    print(df)

# In[4]:


# 載入 木棒台車(CartPole) 遊戲
env = gym.make("CartPole-v1")

# 環境的資訊
print(env.action_space)
print(env.observation_space)
print('observation_space 範圍：')
print(env.observation_space.high)
print(env.observation_space.low)

# In[5]:


# 載入 打磚塊(Breakout) 遊戲
env = gym.make("Breakout-v0")

# 環境的資訊
print(env.action_space)
print(env.observation_space)
print('observation_space 範圍：')
print(env.observation_space.high)
print(env.observation_space.low)

# In[7]:


# 載入 21點(Blackjack-v0) 遊戲
env = gym.make("Blackjack-v1")

# 環境的資訊
print(env.action_space)
print(env.observation_space)

# In[8]:


# 載入 木棒台車(CartPole) 遊戲
env = gym.make("CartPole-v1", render_mode='human')

# 比賽回合結束，重置
observation, info = env.reset()
# 將環境資訊寫入日誌檔
with open("CartPole_random.log", "w", encoding='utf8') as f:
    # 執行 1000 次行動
    for _ in range(1000):
        # 更新畫面
        env.render()
        # 隨機行動
        action = env.action_space.sample()
        # 觸動下一步
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        # 寫入資訊
        f.write(f"action={action}, observation={observation}," + f"reward={reward}, done={done}, info={info}\n")
        # 比賽回合結束，重置
        if done:
            observation, info = env.reset()
env.close()

# In[9]:


# 載入 登山車(MountainCar) 遊戲
env = gym.make("MountainCar-v0", render_mode='human')

# 比賽回合結束，重置
observation, info = env.reset()
# 將環境資訊寫入日誌檔
with open("MountainCar_random.log", "w", encoding='utf8') as f:
    # 執行 1000 次行動
    for _ in range(1000):
        # 更新畫面
        env.render()
        # 隨機行動
        action = env.action_space.sample()
        # 觸動下一步
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        # 寫入資訊
        f.write(f"action={action}, observation={observation}," + f"reward={reward}, done={done}, info={info}\n")
        # 比賽回合結束，重置
        if done:
            observation, info = env.reset()
env.close()
