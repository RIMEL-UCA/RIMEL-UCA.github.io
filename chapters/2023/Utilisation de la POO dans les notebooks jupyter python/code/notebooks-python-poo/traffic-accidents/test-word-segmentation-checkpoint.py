import re, string, random, glob, operator, heapq, functools
from collections import defaultdict
from math import log10
def memo(f):
    "Memoize function f."
    table = {}
    def fmemo(*args):
        if args not in table:
            table[args] = f(*args)
        return table[args]
    fmemo.memo = table
    return fmemo
@memo
def segment(text):
    "Return a list of words that is the best segmentation of text."
    if not text: return []
    candidates = ([first]+segment(rem) for first,rem in splits(text))    
    return max(candidates, key=Pwords)

def splits(text, L=20):
    "Return a list of all possible (first, rem) pairs, len(first)<=L."
    return [(text[:i+1], text[i+1:]) 
            for i in range(min(len(text), L))]

def Pwords(words): 
    "The Naive Bayes probability of a sequence of words."
    return product(Pw(w) for w in words)
#### Support functions (p. 224)

def product(nums):
    "Return the product of a sequence of numbers."
    return functools.reduce(operator.mul, nums, 1)

class Pdist(dict):
    "A probability distribution estimated from counts in datafile."
    def __init__(self, data=[], N=None, missingfn=None):
        for key,count in data:
            self[key] = self.get(key, 0) + int(count)
        self.N = float(N or sum(self.itervalues()))
        self.missingfn = missingfn or (lambda k, N: 1./N)
    def __call__(self, key): 
        if key in self: return self[key]/self.N  
        else: return self.missingfn(key, self.N)

def datafile(name, sep='\t'):
    "Read key,value pairs from file."
    for line in open(name):
        yield line.split(sep)

def avoid_long_words(key, N):
    "Estimate the probability of an unknown word."
    return 10./(N * 10**len(key))
N = 1024908267229 ## Number of tokens
Pw  = Pdist(datafile('count_1w.txt'), N, avoid_long_words)
segment('wheninthecourseofhumaneventsitbecomesnecessary')
segment('inaholeinthegroundtherelivedahobbitnotanastydirtywetholefilledwiththeendsofwormsandanoozysmellnoryetadrybaresandyholewithnothinginittositdownonortoeatitwasahobbitholeandthatmeanscomfort')
samples.iloc[1][1]
import sys
sys.path.insert(0, '../../../')

from classes.wordsegmentation import WordSegmentation

dir_ = "../../../data/v1/NER/"
file_segmentation = dir_+'spanish_count_1w_small_v2_twitter.txt'
segmentation = WordSegmentation(file_segmentation)

import pandas as pd
samples = pd.read_csv('output-words-generator-v2.csv')

result = []
for i in range(len(samples[['original']])):    
    pre = segmentation.segment(samples.iloc[i][1])
    text = ' '.join(pre)
    #result.append([samples.iloc[i][0],text])
    result.append(text)

samples['v3_twitter'] = result
samples.to_csv('output-words-generator-v3-twitter.csv')
result
import pandas as pd
#samples = pd.read_csv('samples_word_segmentation.csv')
samples = pd.read_csv('output-words-generator-v2.csv')
#samples[['original']]
#N = 2495613020 # spanish_count_1w_small
#N = 2497358193 # spanish_count_1w_small_v2 sin acento
N = 2575683488 # spanish_count_1w_small_v2 sin acento con Twitter
Pw  = Pdist(datafile('spanish_count_1w_small_v2_twitter.txt'), N, avoid_long_words)
result = []
for i in range(len(samples[['original']])):
    pre = segment(samples.iloc[i][0])
    text = ' '.join(pre)
    #result.append([samples.iloc[i][0],text])
    result.append(text)
result
samples['v2_twitter'] = result
samples
samples.to_csv('output-words-generator-v2-twitter.csv')
dataframe = pd.DataFrame(result,columns=['original','word_segmentation'])
dataframe
dataframe.to_csv('words-generator.csv')
#N = 1911392132
#Pw  = Pdist(datafile('datos-colombia.txt'), N, avoid_long_words)
#N = 2495613020 # spanish_count_1w_small
N = 2497358193 # spanish_count_1w_small_v3 sin acento
Pw  = Pdist(datafile('spanish_count_1w_small_v2.txt'), N, avoid_long_words)

segment('accionpoetica')
def cPw(word, prev):
    "Conditional probability of word, given previous word."
    try:
        return P2w[prev + ' ' + word]/float(Pw[prev])
    except KeyError:
        return Pw(word)

@memo 
def segment2(text, prev='<S>'): 
    "Return (log P(words), words), where words is the best segmentation." 
    if not text: return 0.0, [] 
    candidates = [combine(log10(cPw(first, prev)), first, *segment2(rem, first)) 
                  for first,rem in splits(text)] 
    return max(candidates) 

def combine(Pfirst, first, Prem, rem): 
    "Combine first and rem results into one (probability, words) pair." 
    return Pfirst+Prem, [first]+rem 
P2w = Pdist(datafile('count_2w.txt'), N)
segment2('inaholeinthegroundtherelivedahobbitnotanastydirtywetholefilledwiththeendsofwormsandanoozysmellnoryetadrybaresandyholewithnothinginittositdownonortoeatitwasahobbitholeandthatmeanscomfort')
segment2('wecouldincorporatemoredataandeitherkeepmoreentriesfromtheunigramorbigramdataorperhapsaddtrigramdata')
segment2('wonderworman')
