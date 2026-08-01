#!/usr/bin/env python
# coding: utf-8

# # PyTorch Geometric(PyG)
# ### 程式修改自 https://pytorch-geometric.readthedocs.io/en/latest/notes/colabs.html

# ## 載入套件

# In[1]:


import torch
from torch_geometric.data import Data
import networkx as nx

# ## 載入內建資料集

# In[2]:


from torch_geometric.datasets import TUDataset

# 載入內建資料
dataset = TUDataset(root='./graph/TUDataset', name='MUTAG')

print()
print(f'Dataset: {dataset}:')
print('====================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

data = dataset[0]  # Get the first graph object.

print()
print(data)
print('=============================================================')

# Gather some statistics about the first graph.
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
print(f'Has isolated nodes: {data.has_isolated_nodes()}')
print(f'Has self-loops: {data.has_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')

# ## 資料分割

# In[3]:


torch.manual_seed(12345)
dataset = dataset.shuffle()   # 洗牌

train_dataset = dataset[:150] # 前 150 筆作為訓練資料
test_dataset = dataset[150:]  # 後 38 筆作為測試資料

print(f'Number of training graphs: {len(train_dataset)}')
print(f'Number of test graphs: {len(test_dataset)}')

# ## 建立 DataLoader

# In[4]:


from torch_geometric.loader import DataLoader

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 顯示每批資料的內容
for step, data in enumerate(train_loader):
    print(f'Step {step + 1}:')
    print('=======')
    print(f'一批內含圖形的個數: {data.num_graphs}')
    print(data)
    print()

# ## 判斷是否使用GPU

# In[5]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ## 定義模型

# In[6]:


from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import GraphConv

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        torch.manual_seed(12345)
        self.conv1 = GraphConv(dataset.num_node_features, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        x = global_mean_pool(x, batch)

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        
        return x

model = GCN(hidden_channels=64).to(device)
print(model)

# ## 模型訓練

# In[7]:


import numpy as np

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

def train():
    model.train()

    for data in train_loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch) 
        loss = criterion(out, data.y)  # 計算損失
        loss.backward()  
        optimizer.step()  
        optimizer.zero_grad()  

def test(loader):
    model.eval()

    correct = 0
    pred_all = np.array([])
    actual_all = np.array([])
    for data in loader:  
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)  
        pred = out.argmax(dim=1)                # 找最大機率
        correct += int((pred == data.y).sum())  # 計算正確個數
        correct_ratio = correct / len(loader.dataset)        # 計算正確比率
        pred_all = np.concatenate((pred_all, pred.cpu().numpy()))
        actual_all = np.concatenate((actual_all, data.y.cpu().numpy()))
    return correct_ratio, pred_all, actual_all


for epoch in range(1, 171):
    train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    print(f'Epoch: {epoch:03d}, 訓練準確率: {train_acc[0]:.4f}, ' +
          f'測試準確率: {test_acc[0]:.4f}')

# ## 混淆矩陣(Confusion matrix)

# In[8]:


from sklearn.metrics import confusion_matrix

confusion_matrix(test_acc[2], test_acc[1])

# ## 降維、視覺化

# In[9]:


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
test_loader_all = DataLoader(dataset[:], batch_size=len(dataset), shuffle=False)
for data in test_loader_all:  
    data = data.to(device)
    out = model(data.x, data.edge_index, data.batch)  
    pred = out.argmax(dim=1)                # 找最大機率
# 繪圖
visualize(out.cpu(), color=data.cpu().y)

# In[ ]:



