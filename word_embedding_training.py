#!/usr/bin/env python
# coding: utf-8

# In[1]:


#DATA MANIPULATION
import pandas as pd
import numpy as np

#EMBEDDING AND PREPROCESSING
from gensim.models import KeyedVectors
import re
import nltk
import gensim
import multiprocessing
import spacy
from sklearn.preprocessing import LabelBinarizer

#TIME CONTROLS
import time

#PLOT
import matplotlib.pyplot as plt
plt.style.use("ggplot")
get_ipython().run_line_magic('matplotlib', 'inline')

#TENSORFLOW AND KERAS
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Input, LSTM, Embedding, Dropout,SpatialDropout1D, Bidirectional, GlobalAveragePooling1D, MaxPooling1D, Flatten
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.preprocessing.text import Tokenizer

import random


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


# In[3]:


def get_weight_matrix(embedding, vocab, seq_len):
    # total vocabulary size plus 0 for unknown words
    vocab_size = len(vocab) + 1
    # define weight matrix dimensions with all 0
    weight_matrix = np.zeros((vocab_size, seq_len))
    # step vocab, store vectors using the Tokenizer's integer mapping
    for word, i in vocab.items():
        weight_matrix[i] = embedding.get(word)
    return weight_matrix


# In[4]:


def cleaning(doc):
    txt = [token.text for token in doc]
    if len(txt) > 2:        
        return re.sub(' +', ' ', ' '.join(txt)).strip()


# In[5]:


def preprocess_string(string, word_vectors):
    unk_string = '<unk>'
    counter = 0
    string = re.sub('[().\\/\-_\+":“0-9]', ' ', str(string)).lower()
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


# In[6]:


def pad_sequence(string, tokenizer):
    encoded_string = tokenizer.texts_to_sequences(string)
    padded_enconded = pad_sequences(encoded_string, maxlen=max_length, padding='post')
    return padded_enconded


# In[7]:


def preprocess_to_predict(string, word_vectors, tokenizer):
    string = preprocess_string(string, word_vectors)    
    padded_sequence = pad_sequence(string, tokenizer)
    
    return padded_sequence


# In[8]:


categories = []
with open('CATEGORIES.txt','r',encoding='utf-8') as f:
    for line in f.read().splitlines():
        categories.append(line)
categories


# In[9]:


cores = multiprocessing.cpu_count()


# In[10]:


df = pd.read_csv('description_list_categorized.csv',sep=';',encoding='latin1').dropna()
df.head()


# In[11]:


raw_embedding = load_embedding('glove_s50.txt')


# In[12]:


txt = [preprocess_string(doc, raw_embedding) for doc in df['Description']]


# In[13]:


df_clean = pd.DataFrame({'clean': txt})
#df_clean = df_clean.dropna().drop_duplicates()
df_clean.shape


# In[14]:


df_clean.tail()


# In[15]:


x_input = df_clean['clean'].values.tolist()
labels = df['Category'].values.tolist()


# In[16]:


temp_df = pd.DataFrame(list(zip(x_input,labels)), columns=['clean','Category']).dropna()
len(temp_df)


# In[17]:


jobs_encoder = LabelBinarizer()
jobs_encoder.fit(temp_df['Category'])
transformed = jobs_encoder.transform(temp_df['Category'])
ohe_df = pd.DataFrame(transformed)
temp_df = pd.concat([temp_df, ohe_df], axis=1).drop(['Category'], axis=1)
temp_df = temp_df.dropna()


# In[18]:


labels_df = temp_df.drop(['clean'], axis=1)
y = labels_df.values
y.shape


# In[19]:


x_input = temp_df['clean'].values.tolist()
len(x_input)


# In[20]:


t = Tokenizer()
t.fit_on_texts(x_input)
vocab_size = len(t.word_index) + 1


# In[21]:


embedding_vectors = get_weight_matrix(raw_embedding, t.word_index, seq_len=50)


# In[22]:


encoded_docs = t.texts_to_sequences(x_input)


# In[23]:


max_length = 200
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')


# In[24]:


padded_docs.shape


# In[25]:


y.shape


# In[26]:


e = Embedding(vocab_size, 50, weights=[embedding_vectors],mask_zero=False, input_length=max_length, trainable=False)


# In[27]:


#ARCHTECTURE #1
try:
    del model
except:
    pass
model = Sequential()
model.add(e)
model.add(MaxPooling1D())
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dense(35, activation='sigmoid'))

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.001),
              metrics=['accuracy'])


# In[29]:


model.summary()


# In[30]:


hist = model.fit([padded_docs], 
                 y, 
                 validation_split=0.2,
                 epochs=200,
                 batch_size=64, 
                 shuffle=True,
                 verbose=2
                )


# In[31]:


history = pd.DataFrame(hist.history)
#plt.figure(figsize=(12,12))

plt.plot(history["loss"], 'r',label='loss')
plt.plot(history["val_loss"], 'b', label='val_loss')
plt.legend()
plt.show()


# In[32]:


history = pd.DataFrame(hist.history)
#plt.figure(figsize=(12,12))

plt.plot(history["accuracy"], 'r',label='acc')
plt.plot(history["val_accuracy"], 'b', label='val_acc')
plt.legend()
plt.show()

