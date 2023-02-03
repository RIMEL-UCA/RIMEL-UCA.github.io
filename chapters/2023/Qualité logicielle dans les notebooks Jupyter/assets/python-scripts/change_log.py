#!/usr/bin/env python
# coding: utf-8

# # Config

# In[1]:


import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as naf

from nlpaug.util import Action

# In[2]:


text = 'The quick fear fox jumps over the lazy'

# # Sentence Augmenter

# In[3]:


aug = nas.ContextualWordEmbsForSentenceAug(model_path='gpt2', include_detail=True)
augmented_data, change_logs = aug.augment('The quick brown fox jumps over the lazy dog.')
print('augmented_data:', augmented_data)

for change_log in reversed(change_logs):
    print('change_log:', change_log)
    break

# # Word Augmenter

# In[4]:


aug = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', include_detail=True)
augmented_data, change_log = aug.augment('The quick brown fox jumps over the lazy dog.')
print('augmented_data:', augmented_data)
print('change_log:', change_log)

# # Character Augmenter

# In[5]:


aug = nac.KeyboardAug(include_detail=True)
augmented_data, change_log = aug.augment('The quick brown fox jumps over the lazy dog.')
print('augmented_data:', augmented_data)
print('change_log:', change_log)

# # Pipeline

# In[6]:


aug = naf.Sequential([
    naw.RandomWordAug(action='substitute', target_words=['A'], name='aug1', include_detail=False),
    naf.Sequential([
        naw.RandomWordAug(action='substitute', target_words=['D'],name='aug2', include_detail=False),
        naw.RandomWordAug(name='aug3', include_detail=True)
    ], include_detail=False, name='pipe2')
], include_detail=True, name='pipe1')

augmented_data, change_log = aug.augment('The quick brown fox jumps over the lazy dog.')
print('augmented_data:', augmented_data)
print('change_log:', change_log)

# In[ ]:




# In[ ]:



