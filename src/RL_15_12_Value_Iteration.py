#!/usr/bin/env python
# coding: utf-8

# # Grid World迷宮之值循環(Value Iteration)

# In[8]:


# 載入相關套件
import numpy as np
from lib.envs.gridworld import GridworldEnv

# In[9]:


# 環境
env = GridworldEnv()

# In[10]:


# 值循環函數
def value_iteration(
    env: GridworldEnv, theta: float = 0.0001, discount_factor: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    # 計算行動值函數
    def one_step_lookahead(state: int, V: np.ndarray) -> np.ndarray:
        A = np.zeros(env.nA)
        for a in range(env.nA):
            for prob, next_state, reward, done in env.P[state][a]:
                A[a] += prob * (reward + discount_factor * V[next_state])
        return A

    # 狀態值函數初始化
    V = np.zeros(env.nS)
    while True:
        delta = 0
        # 更新每個狀態值的函數
        for s in range(env.nS):
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
    policy = np.zeros([env.nS, env.nA])
    for s in range(env.nS):
        # 計算下一步的行動值函數
        A = one_step_lookahead(s, V)
        # 選擇最佳行動
        best_action = np.argmax(A)
        # 永遠採取最佳行動
        policy[s, best_action] = 1.0

    return policy, V


# In[11]:


# 執行值循環
policy, v = value_iteration(env)

# In[12]:


# 顯示結果
print("策略機率分配:")
print(policy)
print("")

print("4x4 策略機率分配 (0=up, 1=right, 2=down, 3=left):")
print(np.reshape(np.argmax(policy, axis=1), env.shape))
print("")

print("4x4 狀態值函數:")
print(v.reshape(env.shape))

# In[6]:


# 驗證答案是否正確
expected_v = np.array([0, -1, -2, -3, -1, -2, -3, -2, -2, -3, -2, -1, -3, -2, -1, 0])
np.testing.assert_array_almost_equal(v, expected_v, decimal=2)

# ## CliffWalkingEnv

# In[7]:


from lib.envs.cliff_walking import CliffWalkingEnv

# 環境
env = CliffWalkingEnv()
# 執行策略循環
policy, v = value_iteration(env)

# In[ ]:


# 顯示結果
print("策略機率分配:")
print(policy)
print("")

print("4x4 策略機率分配 (0=up, 1=right, 2=down, 3=left):")
print(np.reshape(np.argmax(policy, axis=1), env.shape))
print("")

print("4x4 狀態值函數:")
print(v.reshape(env.shape))
