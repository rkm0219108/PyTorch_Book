# get_ipython().system('pip install datasets')

# 任務(Task)
GLUE_TASKS = ["cola", "mnli", "mnli-mm", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]

# 指定任務為 cola
task = "cola"
# 預先訓練模型
model_checkpoint = "distilbert-base-uncased"
# 批量
batch_size = 16

import random
from typing import Any, Dict, cast

import datasets
import evaluate
import numpy as np
import pandas as pd
from IPython.display import HTML, display
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import BestRun, EvalPrediction

actual_task = "mnli" if task == "mnli-mm" else task
# 載入資料集
dataset = datasets.load_dataset("glue", actual_task)
# 載入效能衡量指標
metric = evaluate.load('glue', actual_task)

# 顯示 dataset 資料內容
dataset

# 顯示第一筆內容
dataset["train"][1]




# 隨機抽取資料函數
def show_random_elements(dataset: datasets.Dataset, num_examples: int = 10) -> None:
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset) - 1)
        while pick in picks:
            pick = random.randint(0, len(dataset) - 1)
        picks.append(pick)

    df = pd.DataFrame(dataset[picks])
    for column, typ in dataset.features.items():
        if isinstance(typ, datasets.ClassLabel):
            df[column] = df[column].transform(lambda i: typ.names[i])
    display(HTML(df.to_html()))


df = pd.DataFrame(dataset["train"][:30])
df

# 隨機抽取10筆資料查看
show_random_elements(dataset["train"])

# 顯示效能衡量指標
metric

# 產生兩筆隨機亂數，測試效能衡量指標

fake_preds = np.random.randint(0, 2, size=(64,))
fake_labels = np.random.randint(0, 2, size=(64,))
metric.compute(predictions=fake_preds, references=fake_labels)


# 分詞
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

# 測試兩筆資料，進行分詞
tokenizer("Hello, this one sentence!", "And this sentence goes with it.")

# 任務的資料集欄位
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

# 測試第一筆資料
sentence1_key, sentence2_key = task_to_keys[task]
if sentence2_key is None:
    print(f"Sentence: {dataset['train'][0][sentence1_key]}")
else:
    print(f"Sentence 1: {dataset['train'][0][sentence1_key]}")
    print(f"Sentence 2: {dataset['train'][0][sentence2_key]}")


# 測試 5 筆資料分詞
def preprocess_function(examples: Dict[str, Any]) -> Any:
    if sentence2_key is None:
        return tokenizer(examples[sentence1_key], truncation=True)
    return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True)


preprocess_function(dataset['train'][:5])

# 將所有資料進行分詞
encoded_dataset = dataset.map(preprocess_function, batched=True)


# 載入預先訓練的模型
num_labels = 3 if task.startswith("mnli") else 1 if task == "stsb" else 2
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)

# 定義訓練參數
metric_name = "pearson" if task == "stsb" else "matthews_correlation" if task == "cola" else "accuracy"

args = TrainingArguments(
    "test-glue",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
)


# 定義效能衡量指標計算的函數
def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    predictions, labels = eval_pred
    predictions = cast(np.ndarray, predictions)
    if task != "stsb":
        predictions = np.argmax(predictions, axis=1)
    else:
        predictions = predictions[:, 0]
    result = metric.compute(predictions=predictions, references=labels)
    assert result is not None
    return result


# 定義訓練者(Trainer)物件
validation_key = (
    "validation_mismatched" if task == "mnli-mm" else "validation_matched" if task == "mnli" else "validation"
)

trainer = Trainer(
    model,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset[validation_key],
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

# 模型評估
trainer.evaluate()

# 模型存檔
trainer.save_model('./cola')


# 預測
class SimpleDataset(Dataset):
    def __init__(self, tokenized_texts: Any) -> None:
        self.tokenized_texts = tokenized_texts

    def __len__(self) -> int:
        return len(self.tokenized_texts["input_ids"])

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {k: v[idx] for k, v in self.tokenized_texts.items()}


texts = ["Hello, this one sentence!", "And this sentence goes with it."]
tokenized_texts = tokenizer(texts, padding=True, truncation=True)
new_dataset = SimpleDataset(tokenized_texts)
trainer.predict(new_dataset)

tokenized_texts = tokenizer(
    ["They drank the pub.", "The professor talked us into a stupor."], padding=True, truncation=True
)
new_dataset = SimpleDataset(tokenized_texts)
trainer.predict(new_dataset)

tokenized_texts = tokenizer(["Hello there!", "This is another text"], padding=True, truncation=True)
new_dataset = SimpleDataset(tokenized_texts)
trainer.predict(new_dataset)

# get_ipython().system('pip install optuna')
# get_ipython().system('pip install ray[tune]')


def model_init() -> Any:
    return AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)


trainer = Trainer(
    model_init=model_init,
    args=args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset[validation_key],
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

best_run = trainer.hyperparameter_search(n_trials=10, direction="maximize")
assert isinstance(best_run, BestRun)

best_run

for n, v in best_run.hyperparameters.items():
    setattr(trainer.args, n, v)

trainer.train()
