#!/usr/bin/env python
# coding: utf-8

#
# # 自動撿貨模擬(Simulation)
#
# ### 程式修改自Roberto Sannazzaro, [『How to Automatize a Warehouse Robot』](https://medium.datadriveninvestor.com/get-started-with-q-learning-with-python-how-to-automatize-a-warehouse-robot-7bfae0180301)

# ## Introduction
#
# The use of robotics is constantly expanding in every business sector, automation takes repetitive tasks and aims to automatize them, in order to optimize processes and cut costs.
# In 2012 Amazon purchased Kiva Systems, a company which developed warehouse robots and related technologies, and which was acquired for $775 million. Moreover many other companies implement robots in their warehouses, even robots that can work in 3 dimensions.
#
# <br/><br/>
#
# ![Autonomous warehouse robot](https://media.giphy.com/media/s0urqX40zokIo/giphy.gif)
#
#
#

# ## 倉庫布置圖：
# <img src="https://i.ibb.co/LrTfrgc/warehouse.png">

# ## 載入套件

# In[1]:


import numpy as np

# ## 定義環境(environment)

# In[2]:


# 位置編碼
location_to_state = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11}

# In[3]:


# 行動空間
actions = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

# ## 定義行動限制，假設G點有最高優先度, 故獎勵設為1000

# In[4]:


# 行動限制，1: 可到達，0:不可到達
R = np.array(
    [
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 1000, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
    ]
)

# ## 依TD(1) 演算法更新行動值函數

# In[5]:


# 參數設定
gamma = 0.75
alpha = 0.9

# 行動值函數初始值為 0
Q = np.array(np.zeros([12, 12]))

# 訓練 1000 週期
for i in range(1000):
    # 隨機起始點
    current_state = np.random.randint(0, 12)
    playable_actions = []
    for j in range(12):
        if R[current_state, j] > 0:
            playable_actions.append(j)
    # 任意行動
    next_state = np.random.choice(playable_actions)
    # 更新行動值函數
    TD = R[current_state, next_state] + gamma * Q[next_state, np.argmax(Q[next_state,])] - Q[current_state, next_state]
    Q[current_state, next_state] = Q[current_state, next_state] + alpha * TD

# ## 顯示更新結果：越靠近G點，值函數越高

# In[6]:


import pandas as pd

q_values = pd.DataFrame(Q, columns=[location for location in location_to_state])
s = q_values.round().style.background_gradient(cmap='GnBu')
s

# ## 重新定義行動限制

# In[7]:


R = np.array(
    [
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
    ]
)

# In[8]:


# 定義代碼與位置對照表
state_to_location = {state: location for location, state in location_to_state.items()}
state_to_location

# ## 定義路由訓練函數

# In[9]:


def route(starting_location, ending_location):
    # starting_location, ending_location：起點、終點
    # 位置轉換為代碼
    ending_state = location_to_state[ending_location]
    # 終點有最高優先度
    R_new = np.copy(R)
    R_new[ending_state, ending_state] = 1000

    # 策略評估：訓練 1000 週期
    Q = np.array(np.zeros([12, 12]))
    for i in range(1000):
        current_state = np.random.randint(0, 12)
        playable_actions = []
        for j in range(12):
            if R_new[current_state, j] > 0:
                playable_actions.append(j)
        # 任意行動
        next_state = np.random.choice(playable_actions)
        # 更新行動值函數
        TD = (
            R_new[current_state, next_state]
            + gamma * Q[next_state, np.argmax(Q[next_state,])]
            - Q[current_state, next_state]
        )
        Q[current_state, next_state] = Q[current_state, next_state] + alpha * TD

    # 策略改善：依TD找尋最佳路由
    route = [starting_location]
    next_location = starting_location
    while next_location != ending_location:
        starting_state = location_to_state[starting_location]
        next_state = np.argmax(Q[starting_state,])
        next_location = state_to_location[next_state]
        route.append(next_location)
        starting_location = next_location
    return route


# In[10]:


# 測試 E --> G 最佳路由
route('E', 'G')

# <img src="https://i.ibb.co/VJ1KKcR/warehouse-1.png">

# In[11]:


# 測試 A --> K 最佳路由
route('A', 'K')

# In[12]:


# 3 個點的路由
def best_route(starting_location, intermediary_location, ending_location):
    # 3 個點的路由 = 2 個點的路由 + 2 個點的路由
    return route(starting_location, intermediary_location) + route(intermediary_location, ending_location)[1:]


# In[13]:


# 測試 E --> K --> G 最佳路由
best_route('E', 'K', 'G')

# <img src="https://i.ibb.co/k2HBnyy/warehouse-2.png">

# In[14]:


# 測試 A --> G --> K 最佳路由
initial = "A"
intermediary = "G"
final = "K"
best = best_route(initial, intermediary, final)
print('最佳路由: ')
print(*best, sep=', ')

# ## References
#
# * [An introduction to Q-Learning](https://www.freecodecamp.org/news/an-introduction-to-q-learning-reinforcement-learning-14ac0b4493cc/)
# * [Reinforcement learning](https://medium.com/machine-learning-for-humans/reinforcement-learning-6eacf258b265)
# *[Math of Q-Learning](https://medium.com/datadriveninvestor/math-of-q-learning-python-code-5dcbdc49b6f6)
