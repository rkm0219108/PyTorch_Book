#!/usr/bin/env python
# coding: utf-8

# # PyTorch Geometric(PyG)

# ## 載入套件

# In[2]:


import torch
from torch_geometric.data import Data
import networkx as nx

# ## 建立圖形

# In[3]:


# 定義邊，第一列為起點，第二列為終點，無向圖須雙向設定
edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
# 節點名稱
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

# 建立新圖形
data = Data(x=x, edge_index=edge_index)
data  # 節點及邊均為二維

# In[4]:


# 邊有另一種寫法較直覺，每一元素均為(起點，終點)
edge_index = torch.tensor([[0, 1], [1, 0], [1, 2], [2, 1]], dtype=torch.long)

# 節點名稱
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

# 要加 contiguous
data = Data(x=x, edge_index=edge_index.t().contiguous())
data  # 節點及邊均為二維

# ## 取得圖形資訊

# In[28]:


print(f'是否為有向圖:{data.is_directed()}')
print(f'圖形鍵值:{data.keys}')
print(f'節點名稱:{data["x"]}')
print(f'節點個數:{data.num_nodes}')
print(f'邊名稱:{data["edge_index"]}')
print(f'邊:{data.num_edges}')
print(f'節點屬性個數:{data.num_node_features}')
print(f'未連結的節點個數:{data.has_isolated_nodes()}')
print(f'自我連結的節點個數:{data.has_self_loops()}')
print(f'節點屬性個數:{data.num_node_features}')

# ## 複製到GPU記憶體

# In[8]:


device = torch.device('cuda')
data = data.to(device)

# ## 繪圖

# In[18]:


list(data["edge_index"].cpu().numpy().T)

# ## [PyG/NetworkX格式互轉](https://pytorch-geometric.readthedocs.io/en/latest/modules/utils.html#torch_geometric.utils.to_networkx)

# In[26]:


from torch_geometric.utils.convert import to_networkx


def draw_pyg(Data: Data) -> None:
    G = to_networkx(Data, to_undirected=True)
    # 繪圖
    nx.draw(
        G,
        with_labels=True,
        node_size=1000,
        node_color="#ffff8f",
        width=0.8,
        font_size=14,
    )


draw_pyg(data)

# ## 自訂函數

# In[24]:


def draw_pyg2(data: Data) -> None:
    G = nx.Graph()

    # nodes
    node_list = data["x"].cpu().numpy().reshape(data["x"].shape[0])
    node_list = node_list.astype(int)  # 節點名稱改為整數
    G.add_nodes_from(node_list)

    # edges
    edges = data["edge_index"].cpu().numpy().T
    edge_list = []
    for item in edges:
        edge_list.append((node_list[item[0]], node_list[item[1]]))
    G.add_edges_from(edge_list)

    # 繪圖
    nx.draw(
        G,
        with_labels=True,
        node_size=1000,
        node_color="#ffff8f",
        width=0.8,
        font_size=14,
    )
    # plt.savefig('grap.png')


draw_pyg2(data)

# ## 載入內建資料集

# In[35]:


from torch_geometric.datasets import TUDataset

# 載入內建資料
dataset = TUDataset(root='./graph/ENZYMES', name='ENZYMES')

# 資料集內含的圖形個數
len(dataset)

# In[30]:


# 類別個數, 特徵個數
dataset.num_classes, dataset.num_node_features

# In[31]:


# 讀取第一個圖形
dataset[0]

# ## 隨機抽樣

# In[34]:


# 洗牌
dataset = dataset.shuffle()
# 讀取第一個圖形
dataset[0]

# ## 資料轉換(Data Transform)

# In[37]:


import torch_geometric.transforms as T
from torch_geometric.datasets import ShapeNet

dataset = ShapeNet(root='./graph/ShapeNet')

dataset[0]

# In[36]:


import torch_geometric.transforms as T
from torch_geometric.datasets import ShapeNet

# KNNGraph：使用最近鄰(KNN)演算法，每一點取6個最近的節點
dataset = ShapeNet(root='./graph/ShapeNet', categories=['Airplane'], pre_transform=T.KNNGraph(k=6))

dataset[0]

# In[38]:


15108 / 2518

# In[40]:


# 資料增補：RandomTranslate
dataset = ShapeNet(
    root='./graph/ShapeNet', categories=['Airplane'], pre_transform=T.KNNGraph(k=6), transform=T.RandomTranslate(0.01)
)

dataset[0]

# In[ ]:
