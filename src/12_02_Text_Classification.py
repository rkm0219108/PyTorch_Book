#!/usr/bin/env python
# coding: utf-8

# # 實作本文分類
# ### 程式來自[Text classification with the torchtext library](https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html)

# ## 載入AG News資料集

# In[1]:


from __future__ import annotations

import time
from typing import Callable, Iterable, Iterator

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchtext.data.functional import (
    to_map_style_dataset,
)
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import AG_NEWS
from torchtext.vocab import build_vocab_from_iterator

news = AG_NEWS(split='train')

type(news)

# In[2]:


train_iter = iter(AG_NEWS(split='train'))

# In[3]:


# 取得下一筆資料
next(train_iter)

# ## 判斷GPU是否存在

# In[4]:


device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"

# ## 詞彙表處理

# In[5]:


# 分詞
tokenizer = get_tokenizer('basic_english')


# 建立 Generator 函數
def yield_tokens(data_iter: Iterable[tuple[int, str]]) -> Iterator[list[str]]:
    for _, text in data_iter:
        yield tokenizer(text)  # pyright: ignore[reportReturnType]


# 由 train_iter 建立詞彙字典
vocab = build_vocab_from_iterator(
    yield_tokens(train_iter), specials=["<unk>"]
)  # pyright: ignore[reportArgumentType, reportCallIssue]

# 設定預設的索引值
vocab.set_default_index(vocab["<unk>"])

# In[6]:


# 測試詞彙字典，取得單字的索引值
vocab(['here', 'is', 'an', 'example'])

# ## 參數設定

# In[7]:


EPOCHS = 10  # 訓練週期數
LR = 5  # 學習率
BATCH_SIZE = 64  # 訓練批量
# 取得標註個數
num_class = len(set([label for (label, text) in news]))
vocab_size = len(vocab)
emsize = 64

# ## 定義資料轉換函數

# In[8]:


text_pipeline = lambda x: vocab(tokenizer(x))  # 分詞、取得單字的索引值
label_pipeline = lambda x: int(x) - 1  # 換成索引值

# In[9]:


# 測試資料轉換
print(text_pipeline('here is an example'))
label_pipeline('10')

# ## 建立模型

# In[10]:


class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, num_class: int) -> None:
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)


model = TextClassificationModel(vocab_size, emsize, num_class).to(device)

# ## 定義訓練及評估函數

# In[11]:


# 訓練函數
def train(dataloader: DataLoader) -> None:
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
                '| epoch {:3d} | {:5d}/{:5d} batches '
                '| accuracy {:8.3f}'.format(epoch, idx, len(dataloader), total_acc / total_count)
            )
            total_acc, total_count = 0, 0
            start_time = time.time()


# 評估函數
def evaluate(dataloader: DataLoader) -> float:
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

# In[12]:


# 批次處理
def collate_batch(batch: Iterable[tuple[str, str]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    label_list, text_list, offsets = [], [], [0]
    for _label, _text in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))  # 設定每筆資料的起始位置
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)  # 每筆資料的起始位置累加
    text_list = torch.cat(text_list)
    return label_list.to(device), text_list.to(device), offsets.to(device)


train_iter, test_iter = AG_NEWS()
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

# In[13]:


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
        '| end of epoch {:3d} | time: {:5.2f}s | '
        'valid accuracy {:8.3f} '.format(epoch, time.time() - epoch_start_time, accu_val)
    )
    print('-' * 59)

# ## 模型評估

# In[14]:


print(f'測試資料準確度: {evaluate(test_dataloader):.3f}')

# ## 測試新資料

# In[17]:


# 新聞類別
ag_news_label = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tec"}


# 預測
def predict(text: str, text_pipeline: Callable[[str], list[int]]) -> int:
    with torch.no_grad():
        text_tensor = torch.tensor(text_pipeline(text)).to(device)
        output = model(text_tensor, torch.tensor([0]).to(device))
        return output.argmax(1).item() + 1


# 測試資料
ex_text_str = "MEMPHIS, Tenn. – Four days ago, Jon Rahm was \
    enduring the season’s worst weather conditions on Sunday at The \
    Open on his way to a closing 75 at Royal Portrush, which \
    considering the wind and the rain was a respectable showing. \
    Thursday’s first round at the WGC-FedEx St. Jude Invitational \
    was another story. With temperatures in the mid-80s and hardly any \
    wind, the Spaniard was 13 strokes better in a flawless round. \
    Thanks to his best putting performance on the PGA Tour, Rahm \
    finished with an 8-under 62 for a three-stroke lead, which \
    was even more impressive considering he’d never played the \
    front nine at TPC Southwind."

print(ag_news_label[predict(ex_text_str, text_pipeline)])

# In[ ]:


my_test = open('./nlp_data/news.txt', encoding='utf8').read()
print(ag_news_label[predict(my_test, text_pipeline)])

# In[ ]:
