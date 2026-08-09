#!/usr/bin/env python
# coding: utf-8

# # NetworkX 基礎

# # 繪製圖形

# ## 載入套件

# In[1]:


import itertools

import community as community_louvain
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from networkx.algorithms import approximation as aprx
from networkx.algorithms import community, tree

# ## 建立圖形

# In[2]:


# 建立新圖形
G = nx.Graph()

# ## 加入節點的各種指令

# In[3]:


# 加一個節點
G.add_node(1)

# 一次加 2 個節點
G.add_nodes_from([2, 3])

# 加 2 個節點，並添加顏色屬性
G.add_nodes_from(
    [
        (4, {"color": "red"}),
        (5, {"color": "green"}),
    ]
)

# 產生 0~9 共 10 個節點
H = nx.path_graph(10)
# 將 H 圖形所有節點，併入 G 圖形
G.add_nodes_from(H)

# 將 H 圖形當作一個節點，併入 G 圖形
G.add_node(H)

# 繪製圖形
nx.draw(G, with_labels=True)

# ## 加入邊的各種指令

# In[4]:


# 加邊，連接節點 1 及 2
G.add_edge(1, 2)

# 另一種寫法
e = (2, 3)
G.add_edge(*e)

# 一次加 2 條邊
G.add_edges_from([(1, 2), (1, 3)])

# 繪製圖形
nx.draw(G, with_labels=True)

# In[5]:


# 取得節點及邊個數
G.number_of_nodes(), G.number_of_edges()

# In[6]:


# 取得所有節點及邊
G.nodes(), G.edges()

# In[7]:


# 加邊，連接節點 3 及 4、4 及 5
G.add_edges_from([(3, 4), (4, 5)], color='red')

# 指定節點名稱，取得連接的節點及屬性
G[1], G[4]

# In[8]:


# 移除節點 2
G.remove_node(2)

# 移除邊 1-3
G.remove_edge(1, 3)

# 移除多個節點
G.remove_nodes_from([4, 5])

# 移除多個邊
G.remove_edges_from([(1, 2), (2, 3)])

# In[9]:


# 繪製圖形
nx.draw(G, with_labels=True, cmap=plt.get_cmap('rainbow'))

# In[10]:


# 建立新圖形，同時加節點、邊及屬性
G = nx.Graph([(1, 2, {"color": "yellow"})])

# 繪製圖形
nx.draw(G, with_labels=True, cmap=plt.get_cmap('rainbow'))

# In[11]:


# 清除所有節點及邊
G.clear()

# ## 繪圖

# In[12]:


# 建立圖形
G = nx.Graph()

# 一次加 3 個節點
G.add_nodes_from([1, 2, 3, 4])

# 加邊
G.add_edge(1, 2)
G.add_edge(1, 3)
G.add_edge(2, 4)
G.add_edge(3, 4)

# Plot the graph
nx.draw(G, with_labels=True)

# In[13]:


# 其他屬性
nx.draw(
    G,
    with_labels=True,
    node_size=1000,
    node_color="#ffff8f",
    width=0.8,
    font_size=14,
)

# In[14]:


# save as PNG
plt.savefig("graph/1.png")

# ## 以屬性為邊的寬度

# In[97]:


# 建立圖形
G = nx.Graph()

G.add_edge("1", "2")
G.add_edge("1", "6")
G.add_edges_from([("1", "3"), ("3", "4")])
G.add_edges_from([("1", "5", {"weight": 3}), ("2", "4", {"weight": 5})])

# 權重計算
weights = [1 if G[u][v] == {} else G[u][v]['weight'] for u, v in G.edges()]
# 以權重作為線條的寬度
nx.draw(G, with_labels=True, cmap=plt.get_cmap('rainbow'), width=weights)

# ## 有向圖(Directed Graph)

# In[17]:


DG = nx.DiGraph()
DG.add_edge(2, 1)
DG.add_edge(1, 3)
DG.add_edge(2, 4)
DG.add_edge(1, 2)

# In[18]:


# Plot the graph
nx.draw(DG, with_labels=True, cmap=plt.get_cmap('rainbow'))

# In[19]:


DG = nx.DiGraph()
DG.add_weighted_edges_from([(1, 2, 0.5), (2, 3, 0.75)])
DG.out_degree(1, weight='weight')  # 指定 OutEdgeView 觀看的屬性
list(DG.successors(1))

# In[20]:


# 鄰居，在有向圖會等於 DG.successors
list(DG.neighbors(2))

# In[21]:


# Plot the graph
nx.draw(DG, with_labels=True, cmap=plt.get_cmap('rainbow'))

# ## 載入XML檔案

# In[22]:


clothing_graph = nx.read_graphml("graph/clothing_graph.graphml")
nx.draw_planar(
    clothing_graph,
    arrowsize=12,
    with_labels=True,
    node_size=1000,
    node_color="#ffff8f",
    linewidths=2.0,
    width=1.5,
    font_size=14,
)

# ## 繪製圖形演算法

# In[23]:


nx.draw_circular(
    clothing_graph,
    arrowsize=12,
    with_labels=True,
    node_size=1000,
    node_color="#ffff8f",
    linewidths=2.0,
    width=1.5,
    font_size=14,
)

# In[24]:


nx.draw_kamada_kawai(
    clothing_graph,
    arrowsize=12,
    with_labels=True,
    node_size=1000,
    node_color="#ffff8f",
    linewidths=2.0,
    width=1.5,
    font_size=14,
)

# In[25]:


nx.draw_spring(
    clothing_graph,
    arrowsize=12,
    with_labels=True,
    node_size=1000,
    node_color="#ffff8f",
    linewidths=2.0,
    width=1.5,
    font_size=14,
)

# ## 載入NetworkX套件內建資料集

# In[26]:


# 載入內建資料
G_karate = nx.karate_club_graph()
# 指定佈局，取得節點座標
pos = nx.spring_layout(G_karate)
# 繪製圖形
nx.draw(G_karate, node_color="#ffff8f", with_labels=True, pos=pos)

# In[27]:


# 統計每個節點的連結個數
G_karate.degree()

# In[28]:


G_karate.nodes()

# In[29]:


pos

# In[30]:


G_karate.edges()

# ## 連結個數直方圖

# In[31]:


degree_freq = np.array(nx.degree_histogram(G_karate)).astype('float')
plt.figure(figsize=(10, 8))
plt.stem(degree_freq)  # 繪製垂直線
plt.ylabel("Frequence")
plt.xlabel("Degree")
plt.show()

# ## 尋找最短路徑

# In[32]:


# 傳回每一條最短路徑
nx.shortest_path(G_karate)

# In[33]:


# 指定起點與終點，可傳回最短路徑
nx.shortest_path(G_karate, source=0, target=23)

# ## 使用權重代表距離

# In[8]:


# 圖的邊，weight為距離
edges = [
    (1, 2, {'weight': 4}),
    (1, 3, {'weight': 2}),
    (2, 3, {'weight': 1}),
    (2, 4, {'weight': 5}),
    (3, 4, {'weight': 8}),
    (3, 5, {'weight': 10}),
    (4, 5, {'weight': 2}),
    (4, 6, {'weight': 8}),
    (5, 6, {'weight': 5}),
]
# 邊的名稱
edge_labels = {(1, 2): 4, (1, 3): 2, (2, 3): 1, (2, 4): 5, (3, 4): 8, (3, 5): 10, (4, 5): 2, (4, 6): 8, (5, 6): 5}

# 生成圖
G = nx.Graph()
for i in range(1, 7):
    G.add_node(i)
G.add_edges_from(edges)

# 繪圖
pos = nx.planar_layout(G)
nx.draw(G, node_color="#ffff8f", with_labels=True, pos=pos)

# 在邊顯示權重(weight)
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

# In[10]:


# 起點為 1，到達其他節點的最短路徑
p1 = nx.shortest_path(G, source=1, weight='weight')
p1

# In[4]:


# 起點為 1，終點為 6
p1to6 = nx.shortest_path(G, source=1, target=6, weight='weight')
p1to6

# In[9]:


# 最短路徑的總長度
length = nx.shortest_path_length(G, source=1, target=6, weight='weight')
length

# ## 最小生成樹(Minimum Spanning Tree)

# In[34]:


# 最小生成樹
mst = tree.minimum_spanning_edges(G_karate, algorithm='prim', data=False)
edgelist = list(mst)
sorted(edgelist)  # 排序

# ## 極大團(Maximal Clique)偵測

# In[35]:


max_clique = aprx.max_clique(G_karate)
max_clique

# In[36]:


max_clique_subgraph = G_karate.subgraph(max_clique)
# 繪製圖形
nx.draw_circular(max_clique_subgraph, node_color="#ffff8f", with_labels=True)

# ## 極大團(Maximal Clique)生成

# In[37]:


G = nx.complete_graph(5)  # 5 個節點
nx.draw(G, node_color="#ffff8f", with_labels=True)

# ## 社群偵測(Community Detection)

# In[38]:


# 內建資料，兩個社群，各有 5 個節點，1個相連的節點
G = nx.barbell_graph(5, 1)
nx.draw_kamada_kawai(G, node_color="#ffff8f", with_labels=True)

# ## Girvan/Newman分群法

# In[39]:


communities_generator = community.girvan_newman(G)
top_level_communities = next(communities_generator)
top_level_communities

# In[40]:


next_level_communities = next(communities_generator)
next_level_communities

# ## 使用迴圈產生不同個社群，並顯示分群的衡量指標Modularity

# In[54]:


k = 4  # 分成 2 ~ k+1 群
# Girvan Newman algorithm
comp = community.girvan_newman(G)
for communities in itertools.islice(comp, k):
    print(
        tuple(sorted(c) for c in communities),
        ":\t\t",
        community.modularity(G, communities),
        ":\t",
        community.partition_quality(G, communities),
    )

# ## Louvain 分群法

# In[74]:


G = nx.barbell_graph(5, 1)
community.louvain_communities(G)

# ## Asynchronous Fluid 分群法

# In[92]:


for k in range(2, 6):
    comp = community.asyn_fluidc(G, k)
    print(tuple(sorted(comp)))

# ## python-louvain 套件

# In[ ]:


# !pip install python-louvain

# In[53]:


partition = community_louvain.best_partition(G_karate)
pos = nx.spring_layout(G_karate)
plt.figure(figsize=(8, 8))
plt.axis('off')
nx.draw_networkx_nodes(G_karate, pos, node_size=600, cmap=plt.cm.RdYlBu, node_color=list(partition.values()))
nx.draw_networkx_edges(G_karate, pos, alpha=0.3)
plt.show()

# In[ ]:
