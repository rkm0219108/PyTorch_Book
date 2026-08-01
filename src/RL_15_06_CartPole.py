#!/usr/bin/env python
# coding: utf-8

# # 木棒台車(CartPole)

# In[2]:


# 載入相關套件
import gym
from gym import envs

# ## 隨機行動

# In[3]:


# 參數設定
no = 50        # 比賽回合數

# 載入 木棒台車(CartPole) 遊戲
env = gym.make("CartPole-v1")

# 重置
observation = env.reset()
all_rewards=[] # 每回合總報酬
all_steps=[] # 每回合總步數
total_rewards = 0
total_steps=0

while no > 0:   # 執行 50 比賽回合數
    # 隨機行動
    action = env.action_space.sample() 
    total_steps+=1

    # 觸動下一步
    observation, reward, done, info = env.step(action)
    # 累計報酬
    total_rewards += reward

    # 比賽回合結束，重置
    if done:
        observation = env.reset()
        all_rewards.append(total_rewards)
        all_steps.append(total_steps)
        total_rewards = 0
        total_steps=0
        no-=1

env.close()

# In[4]:


# 顯示執行結果
print('回合\t報酬\t結果')
for i, (rewards, steps) in enumerate(zip(all_rewards, all_steps)):
    result = 'Win' if steps >= 200 else 'Loss'
    print(f'{i}\t{rewards}\t{result}')

# ## 傳統解法

# In[5]:


import math 

# 參數設定
left, right = 0, 1  # 台車行進方向
max_angle = 8       # 偏右8度以上，就往右前進，偏左也是同樣處理

# In[6]:


class Agent:
    # 初始化
    def __init__(self):
        self.direction = left
        self.last_direction=right
        
    # 自訂策略
    def act(self, observation):
        # 台車位置、台車速度、平衡桿角度、平衡桿速度
        cart_position, cart_velocity, pole_angle, pole_velocity = observation
        
        '''
        行動策略：
        1. 設定每次行動採一左一右，盡量不離中心點。
        2. 平衡桿角度偏右8度以上，就往右前進，直到角度偏右小於8度。
        3. 反之，偏左也是同樣處理。
        '''
        if pole_angle < math.radians(max_angle) and \
            pole_angle > math.radians(-max_angle):
            self.direction = (self.last_direction + 1) % 2
        elif pole_angle >= math.radians(max_angle):
            self.direction = right
        else:
            self.direction = left

        self.last_direction = self.direction
        
        return self.direction  

# In[8]:


no = 50        # 比賽回合數

# 載入 木棒台車(CartPole) 遊戲
env = gym.make("CartPole-v1")

# 重置
observation = env.reset()
all_rewards=[] # 每回合總報酬
all_steps=[] # 每回合總步數
total_rewards = 0
total_steps=0

agent = Agent()
while no > 0:   # 執行 50 比賽回合數
    # 行動
    action = agent.act(observation) #env.action_space.sample()
    total_steps+=1

    # 觸動下一步
    observation, reward, done, info = env.step(action)
    # 累計報酬
    total_rewards += reward

    # 比賽回合結束，重置
    if done:
        observation = env.reset()
        all_rewards.append(total_rewards)
        total_rewards = 0
        all_steps.append(total_steps)
        total_steps = 0
        no-=1

env.close()

# In[9]:


# 顯示執行結果
print('回合\t報酬\t結果')
for i, (rewards, steps) in enumerate(zip(all_rewards, all_steps)):
    result = 'Win' if steps >= 200 else 'Loss'
    print(f'{i}\t{rewards}\t{result}')

# ## 以下程式來自：[『From Scratch_ AI Balancing Act in 50 Lines of Python』](https://towardsdatascience.com/from-scratch-ai-balancing-act-in-50-lines-of-python-7ea67ef717)

# In[3]:


import numpy as np

env = gym.make('CartPole-v1')

def play(env, policy):
    observation = env.reset()
    
    done = False
    score = 0
    observations = []
    
    # 訓練5000步
    for _ in range(5000):
        observations += [observation.tolist()] # 記錄歷次狀態
        
        if done: # 回合是否勝負已分
            break
                
        # 行動策略選擇
        outcome = np.dot(policy, observation)
        action = 1 if outcome > 0 else 0
        
        # 觸發下一步
        observation, reward, done, info = env.step(action)
        score += reward

    return score, observations

# In[4]:


np.random.rand(1,4)

# In[5]:


# 訓練 10 回合
max = (0, [], [])
for _ in range(10):
    policy = np.random.rand(1,4) # 產生4個隨機變數 [0, 1)
    score, observations = play(env, policy) # 開始玩
    
    if score > max[0]: # 取最大分數
        max = (score, observations, policy)

print('Max Score', max[0])

# In[6]:


# 最終版本
max = (0, [], [])

for _ in range(100): # 訓練 100 回合    
    policy = np.random.rand(1,4) - 0.5  # 改為 [-0.5, 0.5]
    score, observations = play(env, policy)
    
    if score > max[0]:  # 取最大分數
        max = (score, observations, policy)
        
print('Max Score', max[0])

# ## 以最大分數的policy進行實驗，驗證最佳策略是否有效

# In[11]:


# 取得最佳策略
policy = max[2]
policy

# ## 以最佳策略取代隨機policy，進行 10 回合驗證    

# In[12]:


for _ in range(10): 
    score, observations = play(env, policy)
    print('Score: ', score)

# In[ ]:



