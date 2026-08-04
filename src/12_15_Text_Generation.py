#!/usr/bin/env python
# coding: utf-8

# # 以Transformers套件實作文字生成(Text Generation)功能

# ## 載入相關套件

# In[1]:


from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# ## 載入模型

# In[2]:


text_generator = pipeline("text-generation")

# ## 測試

# In[3]:


print(text_generator("As far as I am concerned, I will", max_length=50, do_sample=False))

# In[4]:


print(text_generator("As far as I am concerned, I will", max_length=50, do_sample=True))

# ## 結合Tokenizer

# In[5]:


# 載入相關套件

# 結合分詞器(Tokenizer)
model = AutoModelForCausalLM.from_pretrained("xlnet-base-cased")
tokenizer = AutoTokenizer.from_pretrained("xlnet-base-cased")

# ## 短文與提示

# In[6]:


# 短文
PADDING_TEXT = """In 1991, the remains of Russian Tsar Nicholas II and his family
(except for Alexei and Maria) are discovered.
The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
remainder of the story. 1883 Western Siberia,
a young Grigori Rasputin is asked by his father and a group of men to perform magic.
Rasputin has a vision and denounces one of the men as a horse thief. Although his
father initially slaps him for making such an accusation, Rasputin watches as the
man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
with people, even a bishop, begging for his blessing. <eod> </s> <eos>"""

# 提示
prompt = "Today the weather is really nice and I am planning on "

# ## 推測答案

# In[7]:


inputs = tokenizer(PADDING_TEXT + prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]

prompt_length = len(tokenizer.decode(inputs[0]))
outputs = model.generate(inputs, max_length=250, do_sample=True, top_p=0.95, top_k=60)
generated = prompt + tokenizer.decode(outputs[0])[prompt_length + 1 :]

print(generated)

# In[ ]:
