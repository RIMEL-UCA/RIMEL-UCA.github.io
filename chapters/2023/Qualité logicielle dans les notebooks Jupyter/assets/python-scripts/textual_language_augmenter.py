#!/usr/bin/env python
# coding: utf-8

# In[1]:


%load_ext autoreload
%autoreload 2
import importlib

# In[2]:


import os
os.environ["MODEL_DIR"] = '../model'

# # Config

# In[11]:


import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as nafc

from nlpaug.util import Action

# # Synonym Augmenter (WordNet, Spanish)

# In[5]:


text = 'Un rápido zorro marrón salta sobre el perro perezoso'
aug = naw.SynonymAug(aug_src='wordnet', lang='spa')
augmented_text = aug.augment(text)
print("Original:")
print(text)
print("Augmented Text:")
print(augmented_text)

# # Word Embeddings Augmenter (word2vec, French)

# In[6]:


augmented_text = aug.augment(text)
print("Original:")
print(text)
print("Augmented Text:")
print(augmented_text)

# # Word Embeddings Augmenter (fasttext, Japanese)

# In[7]:


# https://github.com/taishi-i/nagisa
import nagisa
def tokenizer(text):
    return nagisa.tagging(text).words

text = '速い茶色の狐が怠惰なな犬を飛び越えます'
aug = naw.WordEmbsAug(model_type='fasttext', tokenizer=tokenizer,
                      model_path=os.path.join(os.environ.get("MODEL_DIR"), 'wiki.ja.vec'))
augmented_text = aug.augment(text)
print("Original:")
print(text)
print("Augmented Text:")
print(augmented_text)

# # Contextual Word Embeddings Augmenter (BERT)

# In[12]:


# Augment French by BERT
aug = naw.ContextualWordEmbsAug(model_path='bert-base-multilingual-uncased', aug_p=0.1)
text = "Bonjour, J'aimerais une attestation de l'employeur certifiant que je suis en CDI."
augmented_text = aug.augment(text)
print("Original:")
print(text)
print("Augmented Text:")
print(augmented_text)

# In[9]:


# Augment Japanese by BERT
aug = naw.ContextualWordEmbsAug(model_path='bert-base-multilingual-uncased', aug_p=0.1)
text = '速い茶色の狐が怠惰なな犬を飛び越えます'
augmented_text = aug.augment(text)
print("Original:")
print(text)
print("Augmented Text:")
print(augmented_text)

# In[10]:


# Augment Spanish by BERT
aug = naw.ContextualWordEmbsAug(model_path='bert-base-multilingual-uncased', aug_p=0.1)
text = 'Un rápido zorro marrón salta sobre el perro perezoso'
augmented_text = aug.augment(text)
print("Original:")
print(text)
print("Augmented Text:")
print(augmented_text)

# In[ ]:



