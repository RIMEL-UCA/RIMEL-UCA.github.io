#!/usr/bin/env python
# coding: utf-8

# ## Example of Textual Augmenter Usage<a class="anchor" id="home"></a>:
# * [Character Augmenter](#chara_aug)
#     * [OCR](#ocr_aug)
#     * [Keyboard](#keyboard_aug)
#     * [Random](#random_aug)
# * [Word Augmenter](#word_aug)
#     * [Spelling](#spelling_aug)
#     * [Word Embeddings](#word_embs_aug)
#     * [TF-IDF](#tfidf_aug)
#     * [Contextual Word Embeddings](#context_word_embs_aug)
#     * [Synonym](#synonym_aug)
#     * [Antonym](#antonym_aug)
#     * [Random Word](#random_word_aug)
#     * [Split](#split_aug)
#     * [Back Translatoin](#back_translation_aug)
#     * [Reserved Word](#reserved_aug)
# * [Sentence Augmenter](#sent_aug)
#     * [Contextual Word Embeddings for Sentence](#context_word_embs_sentence_aug)
#     * [Abstractive Summarization](#abst_summ_aug)

# In[5]:


import os
os.environ["MODEL_DIR"] = '../model'

# # Config

# In[1]:


import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as nafc

from nlpaug.util import Action

# In[2]:


text = 'The quick brown fox jumps over the lazy dog .'
print(text)

# # Character Augmenter<a class="anchor" id="chara_aug">
# 
# Augmenting data in character level. Possible scenarios include image to text and chatbot. During recognizing text from image, we need to optical character recognition (OCR) model to achieve it but OCR introduces some errors such as recognizing "o" and "0". `OCRAug` simulate these errors to perform the data augmentation. For chatbot, we still have typo even though most of application comes with word correction. Therefore, `KeyboardAug` is introduced to simulate this kind of errors.

# ### OCR Augmenter<a class="anchor" id="ocr_aug"></a>

# ##### Substitute character by pre-defined OCR error

# In[4]:


aug = nac.OcrAug()
augmented_texts = aug.augment(text, n=3)
print("Original:")
print(text)
print("Augmented Texts:")
print(augmented_texts)

# ### Keyboard Augmenter<a class="anchor" id="keyboard_aug"></a>

# ##### Substitute character by keyboard distance

# In[4]:


aug = nac.KeyboardAug()
augmented_text = aug.augment(text)
print("Original:")
print(text)
print("Augmented Text:")
print(augmented_text)

# ### Random Augmenter<a class="anchor" id="random_aug"></a>

# ##### Insert character randomly

# In[6]:


aug = nac.RandomCharAug(action="insert")
augmented_text = aug.augment(text)
print("Original:")
print(text)
print("Augmented Text:")
print(augmented_text)

# ##### Substitute character randomly

# In[7]:


aug = nac.RandomCharAug(action="substitute")
augmented_text = aug.augment(text)
print("Original:")
print(text)
print("Augmented Text:")
print(augmented_text)

# ##### Swap character randomly

# In[4]:


aug = nac.RandomCharAug(action="swap")
augmented_text = aug.augment(text)
print("Original:")
print(text)
print("Augmented Text:")
print(augmented_text)

# ##### Delete character randomly

# In[8]:


aug = nac.RandomCharAug(action="delete")
augmented_text = aug.augment(text)
print("Original:")
print(text)
print("Augmented Text:")
print(augmented_text)

# # Word Augmenter<a class="anchor" id="word_aug"></a>
# 
# Besides character augmentation, word level is important as well. We make use of word2vec (Mikolov et al., 2013), GloVe (Pennington et al., 2014), fasttext (Joulin et al., 2016), BERT(Devlin et al., 2018) and wordnet to insert and substitute similar word. `Word2vecAug`,  `GloVeAug` and `FasttextAug` use word embeddings to find most similar group of words to replace original word. On the other hand, `BertAug` use language models to predict possible target word. `WordNetAug` use statistics way to find the similar group of words.

# ### Spelling Augmenter<a class="anchor" id="spelling_aug"></a>

# ##### Substitute word by spelling mistake words dictionary

# In[3]:


aug = naw.SpellingAug()
augmented_texts = aug.augment(text, n=3)
print("Original:")
print(text)
print("Augmented Texts:")
print(augmented_texts)

# In[4]:


aug = naw.SpellingAug()
augmented_texts = aug.augment(text, n=3)
print("Original:")
print(text)
print("Augmented Texts:")
print(augmented_texts)

# ### Word Embeddings Augmenter<a class="anchor" id="word_embs_aug"></a>

# ##### Insert word randomly by word embeddings similarity

# In[9]:


# model_type: word2vec, glove or fasttext
aug = naw.WordEmbsAug(
    model_type='word2vec', model_path=model_dir+'GoogleNews-vectors-negative300.bin',
    action="insert")
augmented_text = aug.augment(text)
print("Original:")
print(text)
print("Augmented Text:")
print(augmented_text)

# ##### Substitute word by word2vec similarity

# In[10]:


# model_type: word2vec, glove or fasttext
aug = naw.WordEmbsAug(
    model_type='word2vec', model_path=model_dir+'GoogleNews-vectors-negative300.bin',
    action="substitute")
augmented_text = aug.augment(text)
print("Original:")
print(text)
print("Augmented Text:")
print(augmented_text)

# ### TF-IDF Augmenter<a class="anchor" id="tfidf_aug"></a>

# ##### Insert word by TF-IDF similarity

# In[7]:


aug = naw.TfIdfAug(
    model_path=os.environ.get("MODEL_DIR"),
    action="insert")
augmented_text = aug.augment(text)
print("Original:")
print(text)
print("Augmented Text:")
print(augmented_text)

# ##### Substitute word by TF-IDF similarity

# In[8]:


aug = naw.TfIdfAug(
    model_path=os.environ.get("MODEL_DIR"),
    action="substitute")
augmented_text = aug.augment(text)
print("Original:")
print(text)
print("Augmented Text:")
print(augmented_text)

# ### Contextual Word Embeddings Augmenter<a class="anchor" id="context_word_embs_aug"></a>

# ##### Insert word by contextual word embeddings (BERT, DistilBERT, RoBERTA or XLNet)

# In[15]:


aug = naw.ContextualWordEmbsAug(
    model_path='bert-base-uncased', action="insert")
augmented_text = aug.augment(text)
print("Original:")
print(text)
print("Augmented Text:")
print(augmented_text)

# ##### Substitute word by contextual word embeddings (BERT, DistilBERT, RoBERTA or XLNet)

# In[16]:


aug = naw.ContextualWordEmbsAug(
    model_path='bert-base-uncased', action="substitute")
augmented_text = aug.augment(text)
print("Original:")
print(text)
print("Augmented Text:")
print(augmented_text)

# In[20]:


aug = naw.ContextualWordEmbsAug(
    model_path='distilbert-base-uncased', action="substitute")
augmented_text = aug.augment(text)
print("Original:")
print(text)
print("Augmented Text:")
print(augmented_text)

# In[14]:


aug = naw.ContextualWordEmbsAug(
    model_path='roberta-base', action="substitute")
augmented_text = aug.augment(text)
print("Original:")
print(text)
print("Augmented Text:")
print(augmented_text)

# ### Synonym Augmenter<a class="anchor" id="synonym_aug"></a>

# ##### Substitute word by WordNet's synonym

# In[6]:


aug = naw.SynonymAug(aug_src='wordnet')
augmented_text = aug.augment(text)
print("Original:")
print(text)
print("Augmented Text:")
print(augmented_text)

# ##### Substitute word by PPDB's synonym

# In[8]:


aug = naw.SynonymAug(aug_src='ppdb', model_path=os.environ.get("MODEL_DIR") + 'ppdb-2.0-s-all')
augmented_text = aug.augment(text)
print("Original:")
print(text)
print("Augmented Text:")
print(augmented_text)

# ### Antonym Augmenter<a class="anchor" id="antonym_aug"></a>

# ##### Substitute word by antonym

# In[4]:


aug = naw.AntonymAug()
_text = 'Good boy'
augmented_text = aug.augment(_text)
print("Original:")
print(_text)
print("Augmented Text:")
print(augmented_text)

# ### Random Word Augmenter<a class="anchor" id="random_word_aug"></a>

# ##### Swap word randomly

# In[6]:


aug = naw.RandomWordAug(action="swap")
augmented_text = aug.augment(text)
print("Original:")
print(text)
print("Augmented Text:")
print(augmented_text)

# ##### Delete word randomly

# In[18]:


aug = naw.RandomWordAug()
augmented_text = aug.augment(text)
print("Original:")
print(text)
print("Augmented Text:")
print(augmented_text)

# ##### Delete a set of contunous word will be removed randomly

# In[4]:


aug = naw.RandomWordAug(action='crop')
augmented_text = aug.augment(text)
print("Original:")
print(text)
print("Augmented Text:")
print(augmented_text)

# ### Split Augmenter<a class="anchor" id="split_aug"></a>

# ##### Split word to two tokens randomly

# In[3]:


aug = naw.SplitAug()
augmented_text = aug.augment(text)
print("Original:")
print(text)
print("Augmented Text:")
print(augmented_text)

# ### Back Translation Augmenter<a class="anchor" id="back_translation_aug"></a>

# In[1]:


import nlpaug.augmenter.word as naw

text = 'The quick brown fox jumped over the lazy dog'
back_translation_aug = naw.BackTranslationAug(
    from_model_name='facebook/wmt19-en-de', 
    to_model_name='facebook/wmt19-de-en'
)
back_translation_aug.augment(text)

# In[8]:


# Load models from local path
import nlpaug.augmenter.word as naw

from_model_dir = os.path.join(os.environ["MODEL_DIR"], 'word', 'fairseq', 'wmt19.en-de')
to_model_dir = os.path.join(os.environ["MODEL_DIR"], 'word', 'fairseq', 'wmt19.de-en')

text = 'The quick brown fox jumped over the lazy dog'
back_translation_aug = naw.BackTranslationAug(
    from_model_name=from_model_dir, from_model_checkpt='model1.pt',
    to_model_name=to_model_dir, to_model_checkpt='model1.pt', 
    is_load_from_github=False)
back_translation_aug.augment(text)


# ### Reserved Word Augmenter<a class="anchor" id="reserved_aug"></a>

# In[ ]:


import nlpaug.augmenter.word as naw

text = 'Fwd: Mail for solution'
reserved_tokens = [
    ['FW', 'Fwd', 'F/W', 'Forward'],
]
reserved_aug = naw.ReservedAug(reserved_tokens=reserved_tokens)
augmented_text = reserved_aug.augment(text)

print("Original:")
print(text)
print("Augmented Text:")
print(augmented_text)

# # Sentence Augmentation

# ### Contextual Word Embeddings for Sentence Augmenter<a class="anchor" id="context_word_embs_sentence_aug"></a>

# ##### Insert sentence by contextual word embeddings (GPT2 or XLNet)

# In[6]:


# model_path: xlnet-base-cased or gpt2
aug = nas.ContextualWordEmbsForSentenceAug(model_path='xlnet-base-cased')
augmented_texts = aug.augment(text, n=3)
print("Original:")
print(text)
print("Augmented Texts:")
print(augmented_texts)

# In[7]:


aug = nas.ContextualWordEmbsForSentenceAug(model_path='gpt2')
augmented_text = aug.augment(text)
print("Original:")
print(text)
print("Augmented Text:")
print(augmented_text)

# In[7]:


aug = nas.ContextualWordEmbsForSentenceAug(model_path='gpt2')
augmented_text = aug.augment(text)
print("Original:")
print(text)
print("Augmented Text:")
print(augmented_text)

# In[8]:


aug = nas.ContextualWordEmbsForSentenceAug(model_path='distilgpt2')
augmented_text = aug.augment(text)
print("Original:")
print(text)
print("Augmented Text:")
print(augmented_text)

# ### Abstractive Summarization Augmenter<a class="anchor" id="abst_summ_aug"></a>

# In[7]:


article = """
The history of natural language processing (NLP) generally started in the 1950s, although work can be 
found from earlier periods. In 1950, Alan Turing published an article titled "Computing Machinery and 
Intelligence" which proposed what is now called the Turing test as a criterion of intelligence. 
The Georgetown experiment in 1954 involved fully automatic translation of more than sixty Russian 
sentences into English. The authors claimed that within three or five years, machine translation would
be a solved problem. However, real progress was much slower, and after the ALPAC report in 1966, 
which found that ten-year-long research had failed to fulfill the expectations, funding for machine 
translation was dramatically reduced. Little further research in machine translation was conducted 
until the late 1980s when the first statistical machine translation systems were developed.
"""

aug = nas.AbstSummAug(model_path='t5-base', num_beam=3)
augmented_text = aug.augment(article)
print("Original:")
print(article)
print("Augmented Text:")
print(augmented_text)

# In[ ]:



