#!/usr/bin/env python
# coding: utf-8

# # 實作情緒分析(Sentiment Analysis)

# ## 載入IMDB資料集

# In[25]:


import time
from typing import Any, Callable, Iterable, Iterator

import torch
from torch import nn
from torch.utils.data import DataLoader
from Dataset import random_split
from torchtext.data.functional import (
    to_map_style_dataset,
)
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import IMDB
from torchtext.vocab import build_vocab_from_iterator

imdb = IMDB(split='train')  # pyright: ignore[reportCallIssue]

type(imdb)

# In[90]:


# 取得下一筆資料
train_iter = iter(IMDB(split='train'))  # pyright: ignore[reportCallIssue]

data = next(train_iter)
data

# ## 判斷GPU是否存在

# In[6]:


device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"

# ## 詞彙表處理

# In[7]:


# 分詞
tokenizer = get_tokenizer('basic_english')


# 建立 Generator 函數
def yield_tokens(data_iter: Iterable[tuple[Any, str]]) -> Iterator[list[str]]:
    for _, text in data_iter:
        yield tokenizer(text)  # pyright: ignore[reportReturnType]


# 由 train_iter 建立詞彙字典
vocab = build_vocab_from_iterator(
    yield_tokens(train_iter), specials=["<unk>"]
)  # pyright: ignore[reportArgumentType, reportCallIssue]

# 設定預設的索引值
vocab.set_default_index(vocab["<unk>"])

# In[8]:


# 測試詞彙字典，取得單字的索引值
vocab(['here', 'is', 'an', 'example'])

# ## 參數設定

# In[91]:


EPOCHS = 10  # 訓練週期數
LR = 5  # 學習率
BATCH_SIZE = 64  # 訓練批量
# 取得標註個數
num_class = len(set([label for (label, text) in imdb]))
vocab_size = len(vocab)
emsize = 64
hidden_dim = 16

# ## 定義資料轉換函數

# In[12]:


text_pipeline = lambda x: vocab(tokenizer(x))  # 分詞、取得單字的索引值
label_pipeline = lambda x: 0 if x == 'neg' else 1  # 換成索引值

# In[13]:


# 測試資料轉換
print(text_pipeline('here is an example'))
label_pipeline('pos')

# ## 建立模型

# In[83]:


class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, num_class: int) -> None:
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_class)
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(text, offsets)
        rnn_out, h_out = self.rnn(embedded)
        return self.fc(rnn_out)


model = TextClassificationModel(vocab_size, emsize, num_class).to(device)

# ## 定義訓練及評估函數

# In[84]:


# 訓練函數
def train(dataloader: "DataLoader") -> None:
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 500
    start_time = time.time()

    for idx, (label, text, offsets) in enumerate(dataloader):
        optimizer.zero_grad()
        predicted_label = model(text, offsets)
        loss = criterion(predicted_label, label)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_acc += (predicted_label.argmax(1) == label).sum().item()
        total_count += label.size(0)
        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print(
                f'| epoch {epoch:3d} | {idx:5d}/{len(dataloader):5d} batches '
                f'| accuracy {total_acc / total_count:8.3f}'
            )
            total_acc, total_count = 0, 0
            start_time = time.time()


# 評估函數
def evaluate(dataloader: "DataLoader") -> float:
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (label, text, offsets) in enumerate(dataloader):
            predicted_label = model(text, offsets)
            loss = criterion(predicted_label, label)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc / total_count


# ## 建立DataLoader，逐批訓練

# In[85]:


# 批次處理
def collate_batch(batch: list[tuple[Any, str]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    label_list, text_list, offsets = [], [], [0]
    for _label, _text in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))  # 設定每筆資料的起始位置
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)  # 單字的索引值累加
    text_list = torch.cat(text_list)
    return label_list.to(device), text_list.to(device), offsets.to(device)


train_iter, test_iter = IMDB()  # pyright: ignore[reportCallIssue]
# 轉換為 DataSet
train_dataset = to_map_style_dataset(train_iter)
test_dataset = to_map_style_dataset(test_iter)
# 資料切割，95% 作為訓練資料
num_train = int(len(train_dataset) * 0.95)
split_train_, split_valid_ = random_split(train_dataset, [num_train, len(train_dataset) - num_train])

# 建立DataLoader
train_dataloader = DataLoader(split_train_, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
valid_dataloader = DataLoader(split_valid_, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)

# ## 模型訓練

# In[86]:


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.1)

total_accu = None
for epoch in range(1, EPOCHS + 1):
    epoch_start_time = time.time()
    train(train_dataloader)
    accu_val = evaluate(valid_dataloader)
    if total_accu is not None and total_accu > accu_val:
        scheduler.step()
    else:
        total_accu = accu_val
    print('-' * 59)
    print(
        f'| end of epoch {epoch:3d} | time: {time.time() - epoch_start_time:5.2f}s | '
        f'valid accuracy {accu_val:8.3f} '
    )
    print('-' * 59)

# ## 模型評估

# In[87]:


print(f'測試資料準確度: {evaluate(test_dataloader):.3f}')

# ## 測試新資料

# In[88]:


# 預測
label = {0: '負面', 1: '正面'}


def predict(text: str, text_pipeline: Callable[[str], list[int]]) -> int:
    with torch.no_grad():
        text_tensor = torch.tensor(text_pipeline(text)).to(device)
        output = model(text_tensor, torch.tensor([0]).to(device))
        return output.argmax(1).item()


# 測試資料
my_test = open('nlp_data/imdb_1.txt', encoding='utf8').read()
print(label[predict(my_test, text_pipeline)])

# In[89]:


imdb_iterator = iter(IMDB(split='train'))  # pyright: ignore[reportCallIssue]
acc = 0
for i in range(20000):
    data = next(imdb_iterator)
    acc += 1 if data[0] == predict(data[1], text_pipeline) else 0
print(f'{(acc / 20000):.2%}')

# In[ ]:
