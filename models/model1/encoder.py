import tensorflow as tf
import re
import numpy as np
import os
import time

def gru(units):
   return tf.keras.layers.GRU(units,
                                  return_sequences=True,
                                  return_state=True,
                                  recurrent_initializer='glorot_uniform',
                                  recurrent_activation='sigmoid')

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz, weights_=None, mask=1):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        if mask == 1 and weights_ is not None:
          self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, embeddings_initializer=tf.keras.initializers.Constant(value=weights_), mask_zero=True, trainable=False)
        else:
          self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
          print("masking not done in embedding")
        self.gru = gru(self.enc_units)
          
    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state = hidden)
        return output, state
    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))
