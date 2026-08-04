#!/usr/bin/env python
# coding: utf-8

# # 木棒台車(CartPole) -- 使用 Stable Baselines3 套件

# In[2]:


# get_ipython().system('pip install stable-baselines3[extra]')
# get_ipython().system('pip install pyglet')

# In[1]:


# 載入相關套件
import gymnasium as gym
from stable_baselines3 import A2C

# In[5]:


# 載入 木棒台車(CartPole) 遊戲
env = gym.make("CartPole-v1", render_mode="human")

# 載入 A2C 演算法
model = A2C('MlpPolicy', env, verbose=0)
model.learn(total_timesteps=10000)

# 訓練 10 週期
all_rewards = []  # 每回合總報酬
total_rewards = 0.0
obs, info = env.reset()
no = 0
while no < 10:
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    # 累計報酬
    total_rewards += float(reward)
    env.render()
    if done:
        obs, info = env.reset()
        all_rewards.append(total_rewards)
        total_rewards = 0.0
        no += 1
env.close()

# In[6]:


# 顯示執行結果
print('回合\t報酬\t結果')
for i, rewards in enumerate(all_rewards):
    result = 'Win' if rewards >= 200 else 'Loss'
    print(f'{i}\t{rewards}\t{result}')

# In[ ]:
