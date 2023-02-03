#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas

# In[2]:


text = 'The quick brown fox jumps over the lazy dog .'
print(text)

# # 1 Input and 1 Output
# Use augment function

# In[3]:


aug = nac.KeyboardAug()
augmented_text = aug.augment(text)
print("Original:")
print(text)
print("Augmented Text:")
print(augmented_text)

# # 1 Input and N Output
# Use augment function with n parameter

# In[4]:


aug = nac.KeyboardAug()
augmented_text = aug.augment(text, n=2)
print("Original:")
print(text)
print("Augmented Text:")
print(augmented_text)

# # N Input and N Output
# Use augments function

# In[5]:


texts = [
    'The quick brown fox jumps over the lazy dog .',
    'It is proved that augmentation is one of the anchor to success of computer vision model.'
]

# In[7]:


aug = nac.KeyboardAug()
augmented_text = aug.augment(texts)
print("Original:")
print(texts)
print("Augmented Text:")
print(augmented_text)

# In[ ]:



