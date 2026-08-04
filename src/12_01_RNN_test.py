#!/usr/bin/env python
# coding: utf-8

# # 簡單的RNN實作

# ## 程式參考來源：
# - https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html
# - https://pytorch.org/docs/stable/generated/nn.RNN.html#nn.RNN
# - https://pytorch.org/text/stable/vocab.html
# - https://pytorch.org/text/stable/functional.html#to-tensor
# - https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html
#

# ## 載入相關套件

# In[5]:


import string
from collections import Counter, OrderedDict
from typing import List, Tuple

import numpy as np
import torch
from torch import nn
import torchtext
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import (
    Vocab,
)

# ## 嵌入層測試

# In[3]:


x = torch.LongTensor([[0, 1, 2], [3, 4, 5]])
embeds = nn.Embedding(6, 5)
print(embeds(x))

# In[4]:


embeds.weight

# In[16]:


x = torch.LongTensor([[1, 2, 3], [4, 5, 6]])
embeds = nn.Embedding(7, 5)
print(embeds(x))

# In[53]:


embeds = nn.Embedding(6, 5)
x1 = torch.LongTensor([[0, 1, 2]])
x2 = torch.LongTensor([[3, 4]])
print(embeds(x1))
print(embeds(x2))
embeds.weight

# In[29]:


embeds = nn.Embedding(6, 5, 5)
x1 = torch.LongTensor([[0, 1, 2]])
x2 = torch.LongTensor([[3, 4]])
x3 = torch.LongTensor([[3, 4]])
print(embeds(x1))
print(embeds(x2))
print(embeds(x3))
embeds.weight

# In[2]:


# 測試資料
word_to_ix = {"hello": 0, "world": 1}
# 詞彙表(vocabulary)含2個單字, 轉換為5維的向量
embeds = nn.Embedding(2, 5)
# 測試 hello
lookup_tensor = torch.LongTensor([word_to_ix["hello"]])
hello_embed = embeds(lookup_tensor)
print(hello_embed)

# ## RNN層測試

# In[348]:


torch.randn(5, 3, 10).shape

# In[349]:


# 測試資料
input = torch.randn(5, 10)
# 建立 RNN 物件
rnn = nn.RNN(10, 20, 2)
# RNN 處理
output, hn = rnn(input)
# 顯示輸出及隱藏層的維度
print(output.shape, hn.shape)

# In[350]:


# 測試資料
input = torch.randn(5, 4, 10)
# 建立 RNN 物件
rnn = nn.RNN(10, 20, 2)
# RNN 處理
output, hn = rnn(input)
# 顯示輸出及隱藏層的維度
print(output.shape, hn.shape)

# In[351]:


# 測試資料
input = torch.randn(5, 3, 10)
# 建立 RNN 物件
rnn = nn.RNN(10, 20, 2)
# 隱藏層的輸入
h0 = torch.randn(2, 3, 20)
# RNN 處理
output, hn = rnn(input, h0)
# 顯示輸出及隱藏層的維度
print(output.shape, hn.shape)

# ## 分詞

# In[352]:


tokenizer = get_tokenizer('basic_english')

text = 'Could have done better.'
tokenizer(text)

# ## 詞彙表處理

# In[353]:


# BOW 統計
counter = Counter(tokenizer(text))  # pyright: ignore[reportCallIssue, reportArgumentType]
# 依出現次數降冪排列
sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[1], reverse=True)
# 建立詞彙字典
ordered_dict = OrderedDict(sorted_by_freq_tuples)

# 建立詞彙表物件，並加一個未知單字(unknown)的索引值
vocab_object = torchtext.vocab.vocab(ordered_dict, specials=["<unk>"])
# 設定詞彙表預設值為未知單字(unknown)的索引值
vocab_object.set_default_index(vocab_object["<unk>"])

# 測試
vocab_object['done']

# In[354]:


vocab_object.get_itos()

# In[355]:


vocab_object.__len__()

# In[356]:


string.punctuation

# In[357]:


def create_vocabulary(text_list: List[str]) -> Tuple[Vocab, List[str], List[List[int]]]:
    # 取得標點符號
    stopwords = list(string.punctuation)

    # 去除標點符號
    clean_text_list = []
    clean_tokens_list = []
    for text in text_list:
        tokens = tokenizer(text)
        clean_tokens = []
        for w in tokens:  # pyright: ignore[reportGeneralTypeIssues]
            if w not in stopwords:
                clean_tokens.append(w)
        clean_tokens_list += clean_tokens
        clean_text_list.append(' '.join(clean_tokens))

    # 建立詞彙表物件
    counter = Counter(clean_tokens_list)
    sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    ordered_dict = OrderedDict(sorted_by_freq_tuples)
    vocab_object = torchtext.vocab.vocab(ordered_dict, specials=["<unk>"])
    vocab_object.set_default_index(vocab_object["<unk>"])

    # 將輸入字串轉為索引值：自詞彙表物件查詢索引值
    clean_index_list = []
    for clean_tokens_list in clean_text_list:
        clean_index_list.append(vocab_object.lookup_indices(clean_tokens_list.split(' ')))

    # 輸出 詞彙表物件、去除標點符號的字串陣列、字串陣列的索引值
    return vocab_object, clean_text_list, clean_index_list


# ## 測試

# In[358]:


docs = [
    'Well done!',
    'Good work',
    'Great effort',
    'nice work',
    'Excellent!',
    'Weak',
    'Poor effort!',
    'not good',
    'poor work',
    'Could have done better.',
]

vocab_object, clean_text_list, clean_index_list = create_vocabulary(docs)
vocab_object.get_itos()

# In[359]:


clean_text_list

# In[360]:


clean_index_list

# # 整合以上功能，實作一個簡單的案例，說明相關的處理程序

# ## 建立詞彙表：整理輸入語句，截長補短，使語句長度一致。

# In[361]:


maxlen = 4  # 語句最大字數
# 測試資料
docs = [
    'Well done!',
    'Good work',
    'Great effort',
    'nice work',
    'Excellent!',
    'Weak',
    'Poor effort!',
    'not good',
    'poor work',
    'Could have done better',
]

vocab_object, clean_text_list, clean_index_list = create_vocabulary(docs)

# 若字串過長，刪除多餘單字
clean_index_list = torchtext.functional.truncate(clean_index_list, maxlen)

# 若字串長度不足，後面補 0
while len(clean_index_list[0]) < maxlen:
    clean_index_list[0] += [0]
torchtext.functional.to_tensor(clean_index_list, 0)  # 0:不足補0

# In[362]:


# 測試
embeds = nn.Embedding(vocab_object.__len__(), 5)
X = torchtext.functional.to_tensor(clean_index_list, 0)  # 0:不足補0
embed_output = embeds(X)
print(embed_output.shape)

# ## 加上完全連接層(Linear)

# In[366]:


class RecurrentNetLinear(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, num_class: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim * maxlen, num_class)  # 要乘以 maxlen
        self.embed_dim = embed_dim
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(text)
        out = embedded.reshape(embedded.size(0), -1)  # 轉換成1維
        return self.fc(out)


model = RecurrentNetLinear(vocab_object.__len__(), 10, 1)

# ## 另一種寫法，使用EmbeddingBag

# In[363]:


class RecurrentNetEmbeddingBag(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, num_class: int) -> None:
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, num_class)
        self.embed_dim = embed_dim
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(text)
        return self.fc(embedded)


model = RecurrentNetEmbeddingBag(vocab_object.__len__(), 10, 1)

# In[367]:


# 定義 10 個語句的正面(1)或負面(0)的情緒
y = torch.FloatTensor([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
X = torchtext.functional.to_tensor(clean_index_list, 0)  # 0:不足補0

# 指定優化器、損失函數
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

# 模型訓練
for epoch in range(1000):
    outputs = model.forward(X)  # forward pass
    optimizer.zero_grad()
    loss = criterion(outputs.reshape(-1), y)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        # print(outputs.shape)
        print(f"Epoch: {epoch}, loss: {loss.item():1.5f}")

# In[368]:


# 模型評估
model.eval()
model(X)

# In[369]:


# 測試資料
test_docs = ['great effort', 'well done', 'poor effort']

# 轉成數值
clean_index_list = []
for text in test_docs:
    clean_index_list.append(vocab_object.lookup_indices(text.split(' ')))
while len(clean_index_list[0]) < maxlen:
    clean_index_list[0] += [0]

clean_index_list = torchtext.functional.truncate(clean_index_list, maxlen)
X = torchtext.functional.to_tensor(clean_index_list, 0)  # 0:不足補0
model(X)

# ## 使用詞向量(Word2Vec)

# ## 讀取 GloVe 50維的詞向量，轉換為GloVe 50維的詞向量

# In[302]:


# https://pytorch.org/text/stable/vocab.html#glove
examples = ['great']
vec = torchtext.vocab.GloVe(name='6B', dim=50)
ret = vec.get_vecs_by_tokens(examples, lower_case_backup=True)
ret

# In[303]:


vec.vectors.size()  # pyright: ignore[reportOptionalMemberAccess]

# In[304]:


vec.stoi['great']  # pyright: ignore[reportOptionalSubscript]

# ## Embedding 不需訓練，直接設定嵌入層權重

# In[338]:


class RecurrentNet(nn.Module):
    def __init__(self, weights_matrix: torch.Tensor, num_embeddings: int, embedding_dim: int, num_class: int) -> None:
        super().__init__()
        self.embedding = nn.EmbeddingBag(num_embeddings, embedding_dim)
        # 設定嵌入層權重
        self.embedding.load_state_dict({'weight': weights_matrix})
        self.fc = nn.Linear(embedding_dim, num_class)

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(text)
        return self.fc(embedded)


# ## 測試資料轉換

# In[339]:


docs = [
    'Well done!',
    'Good work',
    'Great effort',
    'nice work',
    'Excellent!',
    'Weak',
    'Poor effort!',
    'not good',
    'poor work',
    'Could have done better',
]

# 將詞彙表轉為詞向量
stopwords = list(string.punctuation)
clean_text_list = []
clean_tokens_list = []
for i, text in enumerate(docs):
    tokens = tokenizer(text.lower())
    clean_tokens = []
    for w in tokens:  # pyright: ignore[reportGeneralTypeIssues]
        if w not in stopwords:
            clean_tokens.append(w)
    clean_tokens_list += clean_tokens
    clean_text_list.append(clean_tokens)
    tokens_vec = vec.get_vecs_by_tokens(clean_tokens)
vocab_list = list(set(clean_tokens_list))
weights_matrix = vec.get_vecs_by_tokens(vocab_list)

# In[340]:


# 定義 10 個語句的正面(1)或負面(0)的情緒
y = torch.FloatTensor([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
X = torch.LongTensor(np.zeros((len(docs), maxlen)))
for i, item in enumerate(clean_text_list):
    for j, token in enumerate(item):
        if token in vocab_list:
            X[i, j] = vocab_list.index(token)
X

# In[341]:


vocab_list

# In[342]:


# 建立模型物件
model = RecurrentNet(torch.FloatTensor(weights_matrix), len(vocab_list), 50, 1)

# 指定優化器、損失函數
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

# 模型訓練
for epoch in range(1000):
    outputs = model.forward(X)  # forward pass
    optimizer.zero_grad()
    loss = criterion(outputs.reshape(-1), y)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        # print(outputs.shape)
        print(f"Epoch: {epoch}, loss: {loss.item():1.5f}")

# In[343]:


# 模型評估
model.eval()
model(X)

# In[344]:


# 測試資料
test_docs = ['great effort', 'well done', 'poor effort']

# 轉成數值
X = torch.LongTensor(np.zeros((len(test_docs), maxlen)))
clean_text_list = []
for i, text in enumerate(test_docs):
    tokens = tokenizer(text.lower())
    clean_tokens = []
    for w in tokens:  # pyright: ignore[reportGeneralTypeIssues]
        if w not in stopwords:
            clean_tokens.append(w)
    clean_text_list.append(clean_tokens)

for i, item in enumerate(clean_text_list):
    for j, token in enumerate(item):
        if token in vocab_list:
            X[i, j] = vocab_list.index(token)

# 預測
model.eval()
model(X)

# ## 將整個詞向量設定為嵌入層權重

# In[295]:


class RecurrentNet2(nn.Module):
    def __init__(self, vec: torch.Tensor, embedding_dim: int, num_class: int) -> None:
        super().__init__()
        # 將整個詞向量設定為嵌入層權重，且嵌入層設為不訓練
        self.embedding = nn.EmbeddingBag.from_pretrained(vec, freeze=True)
        self.fc = nn.Linear(embedding_dim, num_class)

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(text)
        return self.fc(embedded)


model = RecurrentNet2(vec.vectors, vec.dim, 1)

# In[296]:


# 測試資料
docs = [
    'Well done!',
    'Good work',
    'Great effort',
    'nice work',
    'Excellent!',
    'Weak',
    'Poor effort!',
    'not good',
    'poor work',
    'Could have done better',
]

# 轉成數值
X = torch.LongTensor(np.zeros((len(docs), maxlen)))

for i, text in enumerate(docs):
    tokens = tokenizer(text.lower())
    clean_tokens = []
    j = 0
    for w in tokens:  # pyright: ignore[reportGeneralTypeIssues]
        if w not in stopwords:
            # 轉成詞向量索引值
            X[i, j] = vec.stoi[w]  # pyright: ignore[reportOptionalSubscript]
            j += 1
X

# In[297]:


# 指定優化器、損失函數
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

# 模型訓練
for epoch in range(1000):
    outputs = model.forward(X)  # forward pass
    optimizer.zero_grad()
    loss = criterion(outputs.reshape(-1), y)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        # print(outputs.shape)
        print(f"Epoch: {epoch}, loss: {loss.item():1.5f}")

model.eval()
model(X)

# In[299]:


# 測試資料
test_docs = ['great job', 'well done', 'poor job']

# 轉成數值
X = torch.LongTensor(np.zeros((len(test_docs), maxlen)))
for i, text in enumerate(test_docs):
    tokens = tokenizer(text.lower())
    clean_tokens = []
    j = 0
    for w in tokens:  # pyright: ignore[reportGeneralTypeIssues]
        if w not in stopwords:
            X[i, j] = vec.stoi[w]  # pyright: ignore[reportOptionalSubscript]
            j += 1
X

# In[301]:


# 預測
model.eval()
model(X)

# In[ ]:
