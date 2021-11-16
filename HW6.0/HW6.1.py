import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from keras.layers import Dense, Input
from tensorflow import keras
from keras.callbacks import CSVLogger

csv_logger = CSVLogger('HW6.1-log.txt', append=True, separator=';')
#read train data
(train_data, train_label),(test_data,test_label) = mnist.load_data()

train_data = train_data/255

# train with bottleneck = 64
encoder_input = Input(shape = (28,28,1),name = 'img')
x = layers.Flatten()(encoder_input)
encoder_out = layers.Dense(64,activation = 'relu')(x)
encoder = Model(encoder_input,encoder_out,name = 'encoder') 
decoder_input = layers.Dense(64,activation = 'relu')(encoder_out)
x = layers.Dense(784,activation = 'relu')(decoder_input)
decoder_output = layers.Reshape((28,28,1))(x)

opt = keras.optimizers.Adam(lr = 0.001, decay = 1e-6)
autoencoder = Model(encoder_input,decoder_output,name = 'autoencoder')
autoencoder.summary()

autoencoder.compile(opt,loss = 'mse')
history = autoencoder.fit(train_data,train_data,epochs = 20,batch_size = 32,validation_split = 0.3,callbacks=[csv_logger])

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1,len(loss) + 1)
plt.plot(epochs,loss,'bo',label = 'Training Loss')
plt.plot(epochs, val_loss,"b",label = "Validation Loss")
plt.title("Training and validation loss")
plt.legend()
plt.savefig('HW6.1_history.png')
plt.show()


# fashion data 
from tensorflow.keras.datasets import fashion_mnist
(train_data_f, train_label_f),(test_data_f,test_label_f) = mnist.load_data()

train_data_f = train_data_f/255

threshold = 4 * autoencoder.evaluate(train_data,train_data)

predict_result = autoencoder.predict(train_data_f)

count = 0  #number of anomaly number
for i in range(train_data_f.shape[0]):
    if np.mean((train_data_f[i] - predict_result[i])**2) > threshold:
        count += 1
        
print('anomaly propotion; ',count / train_data_f.shape[0])

