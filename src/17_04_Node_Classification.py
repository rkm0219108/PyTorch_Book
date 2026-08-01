#!/usr/bin/env python
# coding: utf-8

# # 論文分類
# ### 程式修改自 https://pytorch-geometric.readthedocs.io/en/latest/notes/colabs.html

# ## 載入套件

# In[1]:


import torch
from torch_geometric.data import Data
import networkx as nx

# ## 載入內建資料集

# In[2]:


from torch_geometric.datasets import Planetoid

# 載入內建資料
dataset = Planetoid(root='./graph/Cora', name='Cora')

# 資料集內含的圖形個數
len(dataset)

# In[3]:


dataset[0].num_nodes, dataset[0].num_edges

# ## 資料集已切割訓練、驗證及測試資料

# In[4]:


data = dataset[0]
# 遮罩的節點個數
data.train_mask.sum().item(), data.val_mask.sum().item(), data.test_mask.sum().item()

# In[5]:


data.train_mask.shape, data.val_mask.shape, data.test_mask.shape

# In[6]:


len(data.y.numpy())

# In[7]:


set(data.y.numpy())

# ## 判斷是否使用GPU

# In[8]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ## 定義模型

# In[9]:


import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

# ## 模型訓練

# In[19]:


model = GCN().to(device)
data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    print(f'Epoch: {epoch+1:03d}, Loss: {loss:.4f}')

# ## 模型評估

# In[20]:


model.eval()
pred = model(data).argmax(dim=1)
correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
acc = int(correct) / int(data.test_mask.sum())
print(f'Accuracy: {acc:.4f}')

# ## 混淆矩陣(Confusion matrix)

# In[12]:


from sklearn.metrics import confusion_matrix

confusion_matrix(data.y[data.test_mask].cpu().numpy(), 
                 pred[data.test_mask].cpu().numpy())

# ## 降維、視覺化

# In[21]:


import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def visualize(h, color):
    # 降維至2個主成份
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())

    plt.figure(figsize=(10,10))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    plt.show()

# 預測    
model.eval()
out = model(data)
# 繪圖
visualize(out.cpu(), color=data.cpu().y)

# In[ ]:



