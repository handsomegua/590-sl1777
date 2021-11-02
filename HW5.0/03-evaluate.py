import keras
import numpy as np
from keras import models
from keras import layers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Flatten
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras import Sequential
from keras.utils.np_utils import *

#modify the data into the textbook format
max_len = 100 
training_samples = 1000
validation_samples = 500
tokenizer = Tokenizer(num_words=10000)


with open('data.txt') as f:
    text = f.read().splitlines() 



with open('label.txt') as f:
    labels = f.read().splitlines() 


tokenizer.fit_on_texts(text)
sequences = tokenizer.texts_to_sequences(text)
data = pad_sequences(sequences,maxlen=max_len)

labels = np.asarray(labels)
print('Shape of data tensor:', data.shape)
print('Shape of labels tensor:', labels.shape)

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
labels = to_categorical(labels,3)



model = keras.models.load_model('lstm_model')
print('The loss and accuracy of lstm are:', model.evaluate(data,labels))

model1 = keras.models.load_model('cnn_model')
print('The loss and accuracy of Conv1d are: ',model1.evaluate(data,labels))


