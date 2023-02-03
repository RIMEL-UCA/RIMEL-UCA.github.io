#!/usr/bin/env python
# coding: utf-8

# # Introduction to language-model-based data augmentation (LAMBADA)
# https://arxiv.org/pdf/1911.03118.pdf
# 
# LAMBADA (Anaby-Tavor et al., 2019) is proposed to generate synthetic data. We follow the approach with modification into nlpaug so that we can generate more data with few lines of code. The following figures show steps of training LAMBADA. We will go through it step by step

# In[1]:


from IPython.display import Image
Image(filename='../res/lambada_algo.png')

# # Step 0: Input Data
# expected column name are "text" and "label"

# In[2]:


import pandas as pd
data = pd.read_csv('../test/res/text/classification.csv')
data

# # Step 1: Train the classifier

# In[3]:


!python ../scripts/lambada/train_cls.py  \
    --train_data_path ../test/res/text/classification.csv \
    --val_data_path ../test/res/text/classification.csv \
    --output_dir ../model/lambada/cls \
    --device cuda \
    --num_epoch 2

# # Step 2a: Processing data for task-adaptive pretraining

# In[4]:


!python ../scripts/lambada/data_processing.py \
    --data_path ../test/res/text/classification.csv \
    --output_dir ../test/res/text

# # Step 2b: Task-adpative pretraining for langauge model

# In[5]:


!source activate py39; python ../scripts/lambada/run_clm.py \
    --tokenizer_name ../model/lambada/cls \
    --model_name_or_path gpt2 \
    --model_type gpt2 \
    --train_file ../test/res/text/mlm_data.txt \
    --output_dir ../model/lambada/gen \
    --do_train \
    --overwrite_output_dir \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --save_steps=10000 \
    --num_train_epochs 2

# # Step 3 ~ 5: Generate synthetic data

# In[6]:


import nlpaug.augmenter.sentence as nas
aug = nas.LambadaAug(model_dir='../model/lambada', threshold=0.3, batch_size=4)

# In[7]:


aug.augment(['LABEL_0', 'LABEL_1', 'LABEL_2', 'LABEL_3'], n=10)

# In[ ]:



