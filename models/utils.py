import csv
import os
import random
import pickle
import numpy as np
  
'''
Opens, loads, reads and parses a csv file.
Returns 2d Numpy array with the file contents.
'''
def readCSV(filepath, delimiter=' ', header=False):
    reader = csv.reader(open(filepath), delimiter=delimiter, quoting=csv.QUOTE_NONE)
    colnames = None
    if header:
        colnames = next(reader)
        colnames = colnames[1: ]
    mat = []
    rownames = []
    for line in reader:
        rownames.append(line[0])
        mat.append(np.array(list(map(float, line[1: ]))))
    print(mat[0])
    return (np.array(mat), rownames, colnames)

'''
Opens and load a pickle dataset.
Returns:
X_all, Y_all, en_word2idx, en_idx2word, en_vocab, hi_word2idx, hi_idx2word, hi_vocab

'''
def load_pickle_dataset(filepath):
    with open(filepath, 'rb') as fp:
        return pickle.load(fp)

'''
Dumps given obj into a file in pickle format
'''
def save_pickle(filepath, obj):
    with open(filepath, 'wb') as fp:
        pickle.dump(obj, fp, -1)


'''
Generates random uniform vector of the given dim
'''
def generate_randvec(dim=50, lower=-0.5, upper=0.5):
    return np.array([random.uniform(lower, upper) for i in range(dim)])

'''
Reads and loads given Glove enbedding file into a dict.
'''
def buildGloveIntoDict(glove_file_path):
    
    #reader = readCSV(glove_file_path, delimiter=' ')    
    reader = csv.reader(open(glove_file_path), delimiter=' ', quoting=csv.QUOTE_NONE)    
    return {line[0]: np.array(list(map(float, line[1: ]))) for line in reader}

def load_glove_embeddings(glove_home, vocab_size, word2inx, emb_dim):
    embedding_weights = np.zeros((vocab_size, emb_dim))
    if emb_dim == 50:
        glove_dict = buildGloveIntoDict(os.path.join(glove_home, 'glove.6B.50d.txt'))
    elif emb_dim == 100:
        glove_dict = buildGloveIntoDict(os.path.join(glove_home, 'glove.6B.100d.txt'))
    elif emb_dim == 200:
        glove_dict = buildGloveIntoDict(os.path.join(glove_home, 'glove.6B.200d.txt'))
    elif emb_dim == 300:       
            glove_dict = buildGloveIntoDict(os.path.join(glove_home, 'glove.6B.300d.txt'))

    for word, index in word2inx.items():
        if word in glove_dict:
            embedding_weights[index, :] = glove_dict[word]
        else:
            embedding_weights[index, :] = generate_randvec(emb_dim)
    return embedding_weights



def max_length(tensor):
    return max(len(t) for t in tensor)
