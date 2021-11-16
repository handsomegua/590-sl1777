from keras import callbacks
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

csv_logger = CSVLogger('HW6.2-log.txt', append=True, separator=';')
#read train data
(train_data, train_label),(test_data,test_label) = mnist.load_data()

train_data = train_data/255
test_data = test_data/255

input = layers.Input(shape=(28, 28, 1))


# https://keras.io/examples/vision/autoencoder/
# Encoder
x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(input)
x = layers.MaxPooling2D((2, 2), padding="same")(x)
x = layers.Conv2D(32, (3, 3), activation="sigmoid", padding="same")(x)
x = layers.MaxPooling2D((2, 2), padding="same")(x)

# Decoder
x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)
x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="softmax", padding="same")(x)
x = layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)

# Autoencoder
autoencoder = Model(input, x)
autoencoder.compile(optimizer="adam", loss="mse")
autoencoder.summary()


history = autoencoder.fit(x=train_data,y=train_data,epochs=10,batch_size=128,shuffle=True,validation_data=(test_data, test_data),callbacks = [csv_logger])

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1,len(loss) + 1)
plt.plot(epochs,loss,'bo',label = 'Training Loss')
plt.plot(epochs, val_loss,"b",label = "Validation Loss")
plt.title("Training and validation loss")
plt.legend()
plt.savefig('HW6.2_history.png')
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

