import codecs
import numpy as np
from collections import Counter

'''
Shuffles given source and target corpus in sync.
Returns list of source and target corpus sentences in a shuffled order.
'''

def shuffle(src, tar):
  p = np.random.permutation(len(src))
  shuffled_src = []
  shuffled_tar = []
  for pi in p:
    shuffled_src.append(src[pi])
    shuffled_tar.append(tar[pi])
  return shuffled_src, shuffled_tar

'''
Checks if the number of non-alphanumeric characters are more than alphanumeric chars
in the given sentence.
'''
def isNonAlpha(row):
  len_sent = len(row) 
  len_nonalpha = sum(not c.isalpha() for c in row)
  if (len_nonalpha > len_sent/2):
    return True
  else:
    return False


'''
Builds a dict of vocab words and their counts in the given list of 
english sentences
'''
def buildEngVocab(sentences):
  envocab_dict = Counter(word.strip(',.%" ;:)(][?!') for sentence in sentences for word in sentence.split())
  return envocab_dict


'''
Builds a dict of vocab words and their counts in the given list of 
Hindi sentences
'''
def buildHinVocab(sentences):
  envocab_dict = Counter(word.strip(',.%" ;:)(।|][?!<>a-zA-Z') for sentence in sentences for word in sentence.split())
  return envocab_dict
 
def printSentence(encoded_sen, idx2word):
  for i in range(len(encoded_sen)):
    print(idx2word[encoded_sen[i]], end=' ')


'''
loads corpus and split it into sentences and return.
'''
def loadParallelCorpus(filePathEn, filePathHi):
  with codecs.open(filePathHi, encoding='utf-8') as f:
    data_hi = f.read()
  with codecs.open(filePathEn, encoding='utf-8') as f:
    data_en = f.read()
  return data_en, data_hi     
