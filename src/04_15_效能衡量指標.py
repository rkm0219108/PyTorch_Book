#!/usr/bin/env python
# coding: utf-8

# # 效能衡量指標(Metrics)

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import precision_score, recall_score, confusion_matrix

# ## 範例1. 混淆矩陣(Confusion Matrix)

# In[2]:


from sklearn.metrics import confusion_matrix

y_true = [0, 0, 0, 1, 1, 1, 1, 1]  # 實際值
y_pred = [0, 1, 0, 1, 0, 1, 0, 1]  # 預測值

# 混淆矩陣(Confusion Matrix)
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
print(f'TP={tp}, FP={fp}, TN={tn}, FN={fn}')

# ## 繪圖

# In[3]:


# 修正中文問題
plt.rcParams['font.sans-serif'] = ['Zhuque Fangsong (technical preview)']
plt.rcParams['axes.unicode_minus'] = False

# 顯示矩陣
fig, ax = plt.subplots(figsize=(2.5, 2.5))

# 1:藍色, 0:白色
ax.matshow([[1, 0], [0, 1]], cmap=plt.cm.Blues, alpha=0.3)

# 標示文字
ax.text(x=0, y=0, s=tp, va='center', ha='center')
ax.text(x=1, y=0, s=fp, va='center', ha='center')
ax.text(x=0, y=1, s=tn, va='center', ha='center')
ax.text(x=1, y=1, s=fn, va='center', ha='center')

plt.xlabel('實際', fontsize=20)
plt.ylabel('預測', fontsize=20)

# x/y 標籤
plt.xticks([0, 1], ['T', 'F'])
plt.yticks([0, 1], ['P', 'N'])
plt.show()

# ## 範例2. 準確率

# In[4]:


print(f'準確率:{accuracy_score(y_true, y_pred)}')
print(f'驗算={(tp+tn) / (tp+tn+fp+fn)}')

# ## 範例3. 精確率

# In[5]:


print(f'精確率:{precision_score(y_true, y_pred)}')
print(f'驗算={(tp) / (tp+fp)}')

# ## 範例4. 召回率

# In[6]:


print(f'召回率:{recall_score(y_true, y_pred)}')
print(f'驗算={(tp) / (tp+fn)}')

# ## 範例5. 依資料檔data/auc_data.csv計算AUC

# ## 讀取資料

# In[7]:


# 讀取資料檔
import pandas as pd

df = pd.read_csv('./data/auc_data.csv')
df

# ## 以Scikit-learn函數計算AUC

# In[8]:


from sklearn.metrics import roc_curve, roc_auc_score, auc

# fpr：假陽率，tpr：真陽率, threshold：各種決策門檻
fpr, tpr, threshold = roc_curve(df['actual'], df['predict'])
print(f'假陽率={fpr}\n\n真陽率={tpr}\n\n決策門檻={threshold}')

# ## 繪製AUC

# In[9]:


# 繪圖
auc1 = auc(fpr, tpr)
## Plot the result
plt.title('ROC/AUC')
plt.plot(fpr, tpr, color='orange', label='AUC = %0.2f' % auc1)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[ ]:
