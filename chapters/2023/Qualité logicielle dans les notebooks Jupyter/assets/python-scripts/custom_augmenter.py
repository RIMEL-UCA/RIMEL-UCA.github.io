#!/usr/bin/env python
# coding: utf-8

# # Config

# In[1]:


import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.flow as naf

from nlpaug.util import Method

# In[2]:


text = 'The quick brown'

# # Character Augmenter

# In[3]:


from nlpaug.augmenter.char import CharAugmenter

# In[4]:


class CustomCharAug(CharAugmenter):
    def __init__(self, name='CustChar_Aug', min_char=2, aug_char_min=1, aug_char_max=10, aug_char_p=0.3,
                 aug_word_min=1, aug_word_max=10, aug_word_p=0.3, tokenizer=None, reverse_tokenizer=None,
                 stopwords=None, verbose=0, stopwords_regex=None):
        super().__init__(
            name=name, action="substitute", min_char=min_char, aug_char_min=aug_char_min, 
                aug_char_max=aug_char_max, aug_char_p=aug_char_p, aug_word_min=aug_word_min, 
                aug_word_max=aug_word_max, aug_word_p=aug_word_p, tokenizer=tokenizer, 
                reverse_tokenizer=reverse_tokenizer, stopwords=stopwords, device='cpu', 
                verbose=verbose, stopwords_regex=stopwords_regex)

        self.model = self.get_model()

    def substitute(self, data):
        results = []
        # Tokenize a text (e.g. The quick brown fox jumps over the lazy dog) to tokens (e.g. ['The', 'quick', ...])
        tokens = self.tokenizer(data)
        # Get target tokens
        aug_word_idxes = self._get_aug_idxes(tokens, self.aug_word_min, self.aug_word_max, self.aug_word_p, Method.WORD)
        
        for token_i, token in enumerate(tokens):
            # Do not augment if it is not the target
            if token_i not in aug_word_idxes:
                results.append(token)
                continue

            result = ''
            chars = self.token2char(token)
            # Get target characters
            aug_char_idxes = self._get_aug_idxes(chars, self.aug_char_min, self.aug_char_max, self.aug_char_p,
                                                 Method.CHAR)
            if aug_char_idxes is None:
                results.append(token)
                continue

            for char_i, char in enumerate(chars):
                if char_i not in aug_char_idxes:
                    result += char
                    continue
                
                # Augmentation: Expect return from 'self.model.predict' is a list of possible outcome. Otherwise, no need to sample
                result += self.sample(self.model.predict(chars[char_i]), 1)[0]

            results.append(result)

        return self.reverse_tokenizer(results)

    def get_model(self):
        return TempModel()
    
# Define your model with "predict" function
class TempModel:
    def __init__(self):
        self.model = {
            'T': ['t'],
            'h': ['H'],
            'e': ['E'],
            'q': ['Q'],
            'u': ['U'],
            'i': ['I'],
            'c': ['C'],
            'k': ['K'],
            'b': ['B'],
            'r': ['R'],
            'o': ['O'],
            'w': ['W'],
            'n': ['n']
        }

    def predict(self, x):
        # you can implement your own logic 
        if x in self.model:
            return self.model[x]
        
        return [x]

aug = CustomCharAug()
aug.augment(text)

# # Word Augmenter

# In[5]:


from nlpaug.augmenter.word import WordAugmenter

# In[6]:


class CustomWordAug(WordAugmenter):
    def __init__(self, name='CustomWord_Aug', aug_min=1, aug_max=10, 
                 aug_p=0.3, stopwords=None, tokenizer=None, reverse_tokenizer=None, 
                 device='cpu', verbose=0, stopwords_regex=None):
        super(CustomWordAug, self).__init__(
            action='insert', name=name, aug_min=aug_min, aug_max=aug_max, 
                 aug_p=aug_p, stopwords=stopwords, tokenizer=tokenizer, reverse_tokenizer=reverse_tokenizer, 
                 device=device, verbose=0, stopwords_regex=stopwords_regex)
        
        self.model = self.get_model()

    def insert(self, data):
        """
        :param tokens: list of token
        :return: list of token
        """
        tokens = self.tokenizer(data)
        results = tokens.copy()

        aug_idexes = self._get_random_aug_idxes(tokens)
        if aug_idexes is None:
            return data
        aug_idexes.sort(reverse=True)

        for aug_idx in aug_idexes:
            new_word = self.sample(self.model, 1)[0]
            results.insert(aug_idx, new_word)

        return self.reverse_tokenizer(results)
    
    def get_model(self):
        return ['Custom1', 'Custom2']
        
        

aug = CustomWordAug()
aug.augment(text)
