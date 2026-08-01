#!/usr/bin/env python
# coding: utf-8

# # FrozenLake之值循環(Value Iteration)

# In[1]:


# 載入相關套件
import numpy as np
import gym

# In[2]:


# 環境
env = gym.make('FrozenLake-v1')
env.reset()

# In[3]:


nS = env.observation_space.n
nA = env.action_space.n

# In[4]:


# 值循環函數
def value_iteration(env, theta=0.0001, discount_factor=1.0):
    # 計算行動值函數
    def one_step_lookahead(state, V):
        A = np.zeros(nA)
        for a in range(nA):
            for prob, next_state, reward, done in env.P[state][a]:
                A[a] += prob * (reward + discount_factor * V[next_state])
        return A

    # 狀態值函數初始化
    V = np.zeros(nS)
    while True:
        delta = 0
        # 更新每個狀態值的函數
        for s in range(nS):
            # 計算下一步的行動值函數
            A = one_step_lookahead(s, V)
            best_action_value = np.max(A)
            # 比較更新前後的差值，取最大值
            delta = max(delta, np.abs(best_action_value - V[s]))
            # 更新狀態值函數
            V[s] = best_action_value        
        # 若最大差值 < 門檻值，則停止評估
        if delta < theta:
            break

    # 一開始採隨機策略，往上/下/左/右走的機率(π)均等
    policy = np.zeros([nS, nA])
    for s in range(nS):
        # 計算下一步的行動值函數
        A = one_step_lookahead(s, V)
        # 選擇最佳行動
        best_action = np.argmax(A)
        # 永遠採取最佳行動
        policy[s, best_action] = 1.0
    
    return policy, V            

# In[5]:


# 執行值循環
policy, v = value_iteration(env)

# In[7]:


# 顯示結果
print("策略機率分配:")
print(policy)
print("")

print("4x4 策略機率分配 (0=up, 1=right, 2=down, 3=left):")
print(np.reshape(np.argmax(policy, axis=1), (int(nS ** 0.5), int(nS ** 0.5))))
print("")

print("4x4 狀態值函數:")
print(v.reshape((int(nS ** 0.5), int(nS ** 0.5))))
