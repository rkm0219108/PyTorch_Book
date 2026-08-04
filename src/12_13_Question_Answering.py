#!/usr/bin/env python
# coding: utf-8

# # 以Transformers套件實作問答(Question Answering)功能

# In[1]:


# 載入相關套件
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

# In[2]:


# 載入模型
nlp = pipeline("question-answering")  # pyright: ignore[reportCallIssue, reportArgumentType]

# In[5]:


# 訓練資料
context = (
    r"Extractive Question Answering is the task of extracting an answer "
    + "from a text given a question. An example of a question answering "
    + "dataset is the SQuAD dataset, which is entirely based on that task. "
    + "If you would like to fine-tune a model on a SQuAD task, you may "
    + "leverage the examples/question-answering/run_squad.py script."
)

# In[6]:


# 測試 2 筆
result = nlp(question="What is extractive question answering?", context=context)
print(
    f"Answer: '{result['answer']}', score: {round(result['score'], 4)}",
    f", start: {result['start']}, end: {result['end']}",
)

print()

result = nlp(question="What is a good example of a question answering dataset?", context=context)
print(
    f"Answer: '{result['answer']}', score: {round(result['score'], 4)}",
    f", start: {result['start']}, end: {result['end']}",
)

# ## 結合Tokenizer

# In[11]:



# 結合分詞器(Tokenizer)
tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

# In[12]:


# 訓練資料
text = r"""
🤗 Transformers (formerly known as pytorch-transformers and pytorch-pretrained-bert) provides general-purpose
architectures (BERT, GPT-2, RoBERTa, XLM, DistilBert, XLNet…) for Natural Language Understanding (NLU) and Natural
Language Generation (NLG) with over 32+ pretrained models in 100+ languages and deep interoperability between
TensorFlow 2.0 and PyTorch.
"""

# In[13]:


# 問題
questions = [
    "How many pretrained models are available in 🤗 Transformers?",
    "What does 🤗 Transformers provide?",
    "🤗 Transformers provides interoperability between which frameworks?",
]

# In[14]:


# 推測答案
for question in questions:
    inputs = tokenizer(question, text, add_special_tokens=True, return_tensors="pt")
    input_ids = inputs["input_ids"].tolist()[0]

    outputs = model(**inputs)
    answer_start_scores = outputs.start_logits
    answer_end_scores = outputs.end_logits

    # Get the most likely beginning of answer with the argmax of the score
    answer_start = torch.argmax(answer_start_scores)
    # Get the most likely end of answer with the argmax of the score
    answer_end = torch.argmax(answer_end_scores) + 1

    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))

    print(f"Question: {question}")
    print(f"Answer: {answer}")

# In[ ]:
