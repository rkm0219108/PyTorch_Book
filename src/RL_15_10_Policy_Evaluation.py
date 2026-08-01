#!/usr/bin/env python
# coding: utf-8

# # Grid World迷宮之策略評估計算

# In[18]:


from IPython.display import Image

Image('./images/grid_world.png')

# In[61]:


# 載入相關套件
import gymnasium as gym
import numpy as np
from lib.envs.gridworld import GridworldEnv

# In[62]:


# 環境
env = GridworldEnv()

# In[63]:


# UP = 0
# RIGHT = 1
# DOWN = 2
# LEFT = 3

# ## 行動轉移機率

# In[64]:


print('{狀態: {行動: [ (轉移機率, 下一個狀態, 獎勵, 是否到達終點), (轉移機率, 下一個狀態, 獎勵, 是否到達終點)]}}')
env.P

# ## 簡單的策略評估函數

# In[78]:


def policy_eval(policy, env, epoch=1, discount_factor=1.0):
    # 狀態值函數初始化
    V = np.zeros(env.nS)
    V1 = np.copy(V)
    no = 0
    while no < epoch:
        # 更新每個狀態值的函數
        for s in range(env.nS):
            v = 0
            # 計算每個行動後的狀態值函數
            for a, action_prob in enumerate(policy[s]):
                # 取得所有可能的下一狀態值
                for prob, next_state, reward, done in env.P[s][a]:
                    # 狀態值函數公式，依照所有可能的下一狀態值函數加總
                    v += action_prob * prob * (reward + discount_factor * V[next_state])
            V1[s] = v
        V = np.copy(V1)
        no += 1
    return np.array(V)


# ## 訓練1週期

# In[79]:


# 隨機策略，機率均等
random_policy = np.ones([env.nS, env.nA]) / env.nA
# 評估
v = policy_eval(random_policy, env, 1)
print("4x4 狀態值函數:")
print(v.reshape(env.shape))

# ## 訓練2週期

# In[80]:


v = policy_eval(random_policy, env, 2)
print("4x4 狀態值函數:")
print(v.reshape(env.shape))

# ## 訓練3週期

# In[81]:


v = policy_eval(random_policy, env, 3)
print("4x4 狀態值函數:")
print(v.reshape(env.shape))

# ## 完整的策略評估函數

# In[82]:


# 策略評估函數
def policy_eval(policy, env, discount_factor=1.0, theta=0.00001):
    # 狀態值函數初始化
    V = np.zeros(env.nS)
    V1 = np.copy(V)
    while True:
        delta = 0
        # 更新每個狀態值的函數
        for s in range(env.nS):
            v = 0
            # 計算每個行動後的狀態值函數
            for a, action_prob in enumerate(policy[s]):
                # 取得所有可能的下一狀態值
                for prob, next_state, reward, done in env.P[s][a]:
                    # 狀態值函數公式，依照所有可能的下一狀態值函數加總
                    v += action_prob * prob * (reward + discount_factor * V[next_state])
            # 比較更新前後的差值，取最大值
            delta = max(delta, np.abs(v - V[s]))
            V1[s] = v
        V = np.copy(V1)
        # 若最大差值 < 門檻值，則停止評估
        if delta < theta:
            break
    return np.array(V)


# In[87]:


# 隨機策略，機率均等
random_policy = np.ones([env.nS, env.nA]) / env.nA
# 評估
v = policy_eval(random_policy, env)

print("4x4 狀態值函數:")
print(v.reshape(env.shape))

# In[85]:


# 驗證答案是否正確
expected_v = np.array([0, -14, -20, -22, -14, -18, -20, -20, -20, -20, -18, -14, -22, -20, -14, 0])
np.testing.assert_array_almost_equal(v, expected_v, decimal=2)

# In[ ]:
