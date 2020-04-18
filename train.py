#!/usr/bin/env python
# coding: utf-8

# In[1]:


#DATA MANIPULATION
import pandas as pd
import numpy as np

#EMBEDDING AND PREPROCESSING
import tensorflow as tf
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding
import gc

#TIME CONTROLS
import time

#PLOT
import matplotlib.pyplot as plt
plt.style.use("ggplot")
#get_ipython().run_line_magic('matplotlib', 'inline')

#TENSORFLOW AND KERAS
from tensorflow import keras
from tensorflow.keras.layers import Dense, Input, LSTM, Embedding, Dropout,SpatialDropout1D, Bidirectional, GlobalAveragePooling1D, MaxPooling1D, Flatten
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization


import random
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
#tf.debugging.set_log_device_placement(True)


# In[2]:


def load_embedding(filename, encoding='utf-8'):
    # load embedding into memory, skip first line
    file = open(filename,'r',encoding=encoding)
    lines = file.readlines()[1:]
    file.close()
    # create a map of words to vectors
    embedding = dict()
    for line in lines:
        parts = line.split()
        # key is string word, value is numpy array for vector
        try:
            embedding[parts[0]] = np.asarray(parts[1:], dtype='float32')
        except:
            pass
    return embedding


def get_weight_matrix(embedding, vocab, seq_len):
    # total vocabulary size plus 0 for unknown words
    vocab_size = len(vocab) + 1
    # define weight matrix dimensions with all 0
    weight_matrix = np.zeros((vocab_size, seq_len))
    # step vocab, store vectors using the Tokenizer's integer mapping
    for word, i in vocab.items():
        weight_matrix[i] = embedding.get(word)
    return weight_matrix

def cleaning(doc):
    txt = [token.text for token in doc]
    if len(txt) > 2:        
        return re.sub(' +', ' ', ' '.join(txt)).strip()

def preprocess_string(string, word_vectors):
    unk_string = '<unk>'
    counter = 0
    string = re.sub('[(,).\\/\-_\+":“0-9]', ' ', str(string)).lower()
    for word in string.split():
        try:
            word_vectors[word]
            string_to_attatch = word
        except:
            string_to_attatch = unk_string
        
        if counter:
            string = string +' '+ string_to_attatch
        else:
            string = string_to_attatch
            counter = 1
    
    return string

def pad_sequence(string, tokenizer):
    encoded_string = tokenizer.texts_to_sequences(string)
    padded_enconded = pad_sequences(encoded_string, maxlen=max_length, padding='post')
    return padded_enconded

def preprocess_to_predict(string, word_vectors, tokenizer):
    string = preprocess_string(string, word_vectors)    
    padded_sequence = pad_sequence(string, tokenizer)
    
    return padded_sequence


# In[3]:


#CONTANTS
MAX_SEQ_LEN = 200 #number of words to consider
INPUT_DIMS = 50 #number of dimensions in GLOVE vector


# In[4]:


df = pd.read_csv(
    'imdb-reviews-pt-br.csv',              
    index_col=0, 
    sep=',',
    encoding='utf-8', 
    dtype=str, 
    quotechar='"').dropna()

raw_embedding = load_embedding('glove_s50.txt')
txt = [preprocess_string(doc, raw_embedding) for doc in df['text_pt']]
df['clean'] = txt
df['sentiment_code'] = np.where(df['sentiment']=='neg', 0, 1)
df.head()


# In[5]:


x_input = df['clean'].values.tolist()
labels = df['sentiment_code'].values


# In[6]:


t = Tokenizer(filters='!"#$%&*+,-./:;=?@[\\]^_`{|}~\t\n')
t.fit_on_texts(x_input)
vocab_size = len(t.word_index) + 1


# In[7]:


encoded_docs = t.texts_to_sequences(x_input)


# In[8]:


padded_docs = pad_sequences(encoded_docs, maxlen=MAX_SEQ_LEN, padding='post')


# In[9]:


padded_docs[0]


# In[10]:


embedding_vectors = get_weight_matrix(raw_embedding, t.word_index, seq_len=50)


# In[11]:


t.word_index['mais']


# In[12]:


embedding_vectors[t.word_index['mais']]


# In[13]:


e = Embedding(vocab_size, INPUT_DIMS, weights=[embedding_vectors], mask_zero=True, input_length=MAX_SEQ_LEN, trainable=False)


# In[14]:


#just to check if everything is in order
masked_output = e(padded_docs[0])
masked_output.shape


# In[15]:


print(masked_output._keras_mask)


# In[16]:


#ARCHTECTURE #3
try:
    del model
except:
    pass
model = Sequential([
    e,
    Bidirectional(LSTM(64,  return_sequences=True)),
    Bidirectional(LSTM(32)),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1)

])

model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])


# In[17]:


model.summary()


# In[18]:


gc.collect()


# In[ ]:


with tf.device('/device:GPU:0'):

    hist = model.fit(padded_docs, 
                     labels, 
                     validation_split=0.2,
                     epochs=20,
                     batch_size=32, 
                     shuffle=True,
                     verbose=1
    )


# In[ ]:


history = pd.DataFrame(hist.history)
#plt.figure(figsize=(12,12))

plt.plot(history["loss"], 'r',label='loss')
plt.plot(history["val_loss"], 'b', label='val_loss')
plt.legend()
plt.show()


# In[ ]:


history = pd.DataFrame(hist.history)
#plt.figure(figsize=(12,12))

plt.plot(history["accuracy"], 'r',label='acc')
plt.plot(history["val_accuracy"], 'b', label='val_acc')
plt.legend()
plt.show()


# In[ ]:


model.save_weights('easy_checkpoint')

