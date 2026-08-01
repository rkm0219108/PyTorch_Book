#!/usr/bin/env python
# coding: utf-8

# # 效能調校(Fine Tuning)作法，修改自
# https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/text_classification.ipynb

# In[4]:


! pip install datasets 

# ## 參數設定

# ## 定義GLUE所有任務(Task)

# In[1]:


GLUE_TASKS = ["cola", "mnli", "mnli-mm", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]

# ## 指定任務為 cola

# In[2]:


task = "cola"
# 預先訓練模型
model_checkpoint = "distilbert-base-uncased"
# 批量
batch_size = 16

# ## 載入資料集、效能衡量指標

# In[ ]:


import datasets

actual_task = "mnli" if task == "mnli-mm" else task
# 載入資料集
dataset = datasets.load_dataset("glue", actual_task)
# 載入效能衡量指標
metric = datasets.load_metric('glue', actual_task)

# ### dataset 資料型態為 [`DatasetDict`](https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasetdict)

# In[5]:


dataset

# ## 顯示第一筆內容

# In[24]:


dataset["train"][0]

# ## 定義隨機抽取數據函數

# In[7]:


import random
import pandas as pd
from IPython.display import display, HTML

# 隨機抽取資料函數
def show_random_elements(dataset, num_examples=10):
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)
    
    df = pd.DataFrame(dataset[picks])
    for column, typ in dataset.features.items():
        if isinstance(typ, datasets.ClassLabel):
            df[column] = df[column].transform(lambda i: typ.names[i])
    display(HTML(df.to_html()))

# ## 查看前30筆資料

# In[49]:


df = pd.DataFrame(dataset["train"][:30])
df

# ## 隨機抽取10筆資料查看

# In[8]:


show_random_elements(dataset["train"])

# The metric is an instance of [`datasets.Metric`](https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Metric):

# ## 顯示效能衡量指標

# In[9]:


metric

# ## 產生兩筆隨機亂數，測試效能衡量指標

# In[10]:


import numpy as np

fake_preds = np.random.randint(0, 2, size=(64,))
fake_labels = np.random.randint(0, 2, size=(64,))
metric.compute(predictions=fake_preds, references=fake_labels)

# Note that `load_metric` has loaded the proper metric associated to your task, which is:
# 
# - for CoLA: [Matthews Correlation Coefficient](https://en.wikipedia.org/wiki/Matthews_correlation_coefficient)
# - for MNLI (matched or mismatched): Accuracy
# - for MRPC: Accuracy and [F1 score](https://en.wikipedia.org/wiki/F1_score)
# - for QNLI: Accuracy
# - for QQP: Accuracy and [F1 score](https://en.wikipedia.org/wiki/F1_score)
# - for RTE: Accuracy
# - for SST-2: Accuracy
# - for STS-B: [Pearson Correlation Coefficient](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient) and [Spearman's_Rank_Correlation_Coefficient](https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient)
# - for WNLI: Accuracy
# 
# so the metric object only computes the one(s) needed for your task.

# ## 分詞

# In[11]:


from transformers import AutoTokenizer

# 分詞
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

# You can directly call this tokenizer on one sentence or a pair of sentences:

# ## 測試兩筆資料，進行分詞

# In[12]:


tokenizer("Hello, this one sentence!", "And this sentence goes with it.")

# ## 定義任務的資料集欄位

# In[13]:


task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mnli-mm": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

# ## 測試第一筆資料

# In[14]:


sentence1_key, sentence2_key = task_to_keys[task]
if sentence2_key is None:
    print(f"Sentence: {dataset['train'][0][sentence1_key]}")
else:
    print(f"Sentence 1: {dataset['train'][0][sentence1_key]}")
    print(f"Sentence 2: {dataset['train'][0][sentence2_key]}")

# ## 測試 5 筆資料分詞

# In[16]:


def preprocess_function(examples):
    if sentence2_key is None:
        return tokenizer(examples[sentence1_key], truncation=True)
    return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True)

preprocess_function(dataset['train'][:5])

# To apply this function on all the sentences (or pairs of sentences) in our dataset, we just use the `map` method of our `dataset` object we created earlier. This will apply the function on all the elements of all the splits in `dataset`, so our training, validation and testing data will be preprocessed in one single command.

# In[17]:


# 將所有資料進行分詞
encoded_dataset = dataset.map(preprocess_function, batched=True)

# ## 效能微調(Fine tuning)，先加載預先訓練的模型

# In[18]:


from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

# 載入預先訓練的模型
num_labels = 3 if task.startswith("mnli") else 1 if task=="stsb" else 2
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)

# ## 定義訓練參數，可參閱 [`TrainingArguments`](https://huggingface.co/transformers/main_classes/trainer.html#transformers.TrainingArguments)

# In[19]:


metric_name = "pearson" if task == "stsb" else "matthews_correlation" \
                        if task == "cola" else "accuracy"

args = TrainingArguments(
    "test-glue",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
)

# ## 定義效能衡量指標計算的函數

# In[20]:


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if task != "stsb":
        predictions = np.argmax(predictions, axis=1)
    else:
        predictions = predictions[:, 0]
    return metric.compute(predictions=predictions, references=labels)

# ## 定義訓練者(Trainer)物件

# In[21]:


validation_key = "validation_mismatched" if task == "mnli-mm" else \
                 "validation_matched" if task == "mnli" else "validation"

trainer = Trainer(
    model,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset[validation_key],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# ## 模型訓練

# In[22]:


trainer.train()

# ## 模型評估

# In[23]:


trainer.evaluate()

# ## 模型存檔

# In[27]:


trainer.save_model('./cola')

# ## 預測

# In[57]:


class SimpleDataset:
    def __init__(self, tokenized_texts):
        self.tokenized_texts = tokenized_texts
    
    def __len__(self):
        return len(self.tokenized_texts["input_ids"])
    
    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.tokenized_texts.items()}

texts = ["Hello, this one sentence!", "And this sentence goes with it."]    
tokenized_texts = tokenizer(texts, padding=True, truncation=True)
new_dataset = SimpleDataset(tokenized_texts)
trainer.predict(new_dataset)

# In[51]:


tokenized_texts = tokenizer(["They drank the pub.", "The professor talked us into a stupor."]
                            , padding=True, truncation=True)
new_dataset = SimpleDataset(tokenized_texts)
trainer.predict(new_dataset)

# In[56]:


tokenized_texts = tokenizer(["Hello there!", "This is another text"]
                            , padding=True, truncation=True)
new_dataset = SimpleDataset(tokenized_texts)
trainer.predict(new_dataset)

# To see how your model fared you can compare it to the [GLUE Benchmark leaderboard](https://gluebenchmark.com/leaderboard).

# ## Hyperparameter search

# The `Trainer` supports hyperparameter search using [optuna](https://optuna.org/) or [Ray Tune](https://docs.ray.io/en/latest/tune/). For this last section you will need either of those libraries installed, just uncomment the line you want on the next cell and run it.

# In[ ]:


! pip install optuna
! pip install ray[tune]

# During hyperparameter search, the `Trainer` will run several trainings, so it needs to have the model defined via a function (so it can be reinitialized at each new run) instead of just having it passed. We jsut use the same function as before:

# In[ ]:


def model_init():
    return AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)

# And we can instantiate our `Trainer` like before:

# In[ ]:


trainer = Trainer(
    model_init=model_init,
    args=args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset[validation_key],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# The method we call this time is `hyperparameter_search`. Note that it can take a long time to run on the full dataset for some of the tasks. You can try to find some good hyperparameter on a portion of the training dataset by replacing the `train_dataset` line above by:
# ```python
# train_dataset = encoded_dataset["train"].shard(index=1, num_shards=10) 
# ```
# for 1/10th of the dataset. Then you can run a full training on the best hyperparameters picked by the search.

# In[ ]:


best_run = trainer.hyperparameter_search(n_trials=10, direction="maximize")

# The `hyperparameter_search` method returns a `BestRun` objects, which contains the value of the objective maximized (by default the sum of all metrics) and the hyperparameters it used for that run.

# In[ ]:


best_run

# You can customize the objective to maximize by passing along a `compute_objective` function to the `hyperparameter_search` method, and you can customize the search space by passing a `hp_space` argument to `hyperparameter_search`. See this [forum post](https://discuss.huggingface.co/t/using-hyperparameter-search-in-trainer/785/10) for some examples.
# 
# To reproduce the best training, just set the hyperparameters in your `TrainingArgument` before creating a `Trainer`:

# In[ ]:


for n, v in best_run.hyperparameters.items():
    setattr(trainer.args, n, v)

trainer.train()

# Don't forget to [upload your model](https://huggingface.co/transformers/model_sharing.html) on the [ߤ砍odel Hub](https://huggingface.co/models). You can then use it only to generate results like the one shown in the first picture of this notebook!

# In[ ]:



