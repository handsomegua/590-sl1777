# read the data
from keras import preprocessing 
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import codecs
import re
from nltk.tokenize import sent_tokenize
import os
import numpy as np
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

x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples: training_samples + validation_samples]
y_val = labels[training_samples: training_samples + validation_samples]


y_train = to_categorical(y_train,3)
y_val = to_categorical(y_val,3)



# CNN  
from keras import regularizers
from keras import models 
from keras import layers
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense
from keras.callbacks import CSVLogger
embedding_dim = 100

csv_logger = CSVLogger('log_cnn.txt', append=True, separator=';')
model = Sequential()
model.add(Embedding(10000, embedding_dim, input_length=max_len))
model.add(layers.Conv1D(filters= 2, kernel_size= 4,activation= 'softmax',strides=1,input_shape = (100,1000)))

model.add(layers.Dense(16, activation='sigmoid',
                       kernel_regularizer=regularizers.l2(l=0.05)))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='sigmoid'))
model.summary()



# train and save the model fitting procedure into log.txt
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['acc'])
history = model.fit(x_train, y_train,
                    epochs=40,
                    batch_size=32,
                    validation_data=(x_val, y_val),callbacks=[csv_logger])
model.save('cnn_model')


# visualization for cnn 
import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')

plt.title('Training and validation accuracy under CNN model')
plt.legend()
plt.savefig('cnn_acc.png')
plt.show()
plt.figure()

plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss under CNN model')
plt.legend()
plt.savefig('cnn_loss.png')




# lstm
from keras.layers import LSTM,SpatialDropout1D
model = Sequential()
model.add(Embedding(10000, embedding_dim, input_length=max_len))
model.add(LSTM(32,kernel_regularizer=regularizers.l1_l2(0.02,0.05)))
# model.add(Dense(32, activation='softmax'))
model.add(Dense(3,activation='softmax'))
rnn_logger = CSVLogger('log_rnn.txt', append=True, separator=';')
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['acc'])
history = model.fit(x_train, y_train,
                    epochs=20,
                    batch_size=32,
                    validation_data=(x_val, y_val),callbacks=[rnn_logger])
model.save('lstm_model')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')

plt.title('Training and validation accuracy under LSTM')
plt.legend()
plt.savefig('lstm_acc.png')
plt.show()
plt.figure()

plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss under LSTM')
plt.legend()
plt.savefig('lstm_loss.png')
