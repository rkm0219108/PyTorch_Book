#!/usr/bin/env python
# coding: utf-8

# # 簡易的Grid World迷宮之策略評估計算

# In[18]:


from IPython.display import Image
Image('./images/grid_world.png')

# In[19]:


# 載入相關套件
import numpy as np
import random

# In[37]:


# 遊戲參數
gridSize = 4     # 4x4 格
rewardValue = -1 # 獎勵
terminationStates = [[0,0], [gridSize-1, gridSize-1]] # 兩個終點
actions = [[-1, 0], [1, 0], [0, 1], [0, -1]] # 行動空間

# In[38]:


# 行動及獎勵
def actionValue(initialPosition,action):
    if initialPosition in terminationStates: # 到達終點
        finalPosition = initialPosition      # 不移動，留在原來位置
        reward=0  # 獎勵為 0
    else:  # 不在終點，更新位置
        finalPosition = np.array(initialPosition) + np.array(action)
        reward= rewardValue  # 獎勵為 -1

    # 超出邊界時，仍退回原來位置
    if -1 in finalPosition or gridSize in finalPosition:
        finalPosition = initialPosition  # 退回原來位置
        reward= rewardValue  # 獎勵為 -1
    
    return finalPosition, reward

# ## 簡單的策略評估函數

# In[46]:


# numIterations：訓練週期
# gamma：折扣因子
# valueMap：狀態值函數初始值
def policy_evaluation(numIterations,gamma,valueMap):
    valueMap1 = np.copy(valueMap)
    for i in range(numIterations):
        for state in states:        # 更新每一週期的狀態值函數
            weightedRewards=0       # 平均獎勵
            for action in actions:  # 計算每一行動的狀態值函數
                finalPosition,reward = actionValue(state,action)
                # 貝爾曼方程式(Bellman Equation) V(s)
                weightedRewards += 1/4 * (reward + 
                     gamma * valueMap[finalPosition[0],finalPosition[1]])
            valueMap1[state[0],state[1]]=weightedRewards
        valueMap = np.copy(valueMap1)
        print(valueMap)

# ## 訓練3週期

# In[47]:


valueMap = np.zeros((gridSize, gridSize))  # 狀態值函數初始值 = 0
# 狀態位置 (x, y)
states = [[i, j] for i in range(gridSize) for j in range(gridSize)]
policy_evaluation(3,1,valueMap) # 訓練3週期，折扣因子=1

# In[ ]:



