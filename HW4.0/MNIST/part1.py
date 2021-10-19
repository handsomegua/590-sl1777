#Flags 
CNN = True 
DFF_ANN = False 
folds = 5 
model_save = False
epochs = 5
Loss = 'categorical_crossentropy'
optimize = 'rmsprop'
Metric = ['accuracy']
#data 
default = "mnist_fashion"
model_name = "ourmodel"
model_load = False
model_save = False

#MODIFIED FROM CHOLLETT P120 
import matplotlib.pyplot as plt
from keras import layers 
from keras import models
import numpy as np
import warnings
warnings.filterwarnings("ignore")

#-------------------------------------
#BUILD MODEL SEQUENTIALLY (LINEAR STACK)
#-------------------------------------
#initialize the model
model = models.Sequential()
#Add a  32 convnet for 3*3 , activation = relu, input image shape 150*150 * 3 color
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
#Add  max_pooling layer
model.add(layers.MaxPooling2D((2, 2)))
#Add a convnet for 3*3 and 64 of them, activation = relu
model.add(layers.Conv2D(64, (3, 3), activation='relu')) 
#Add a max_pooling layer
model.add(layers.MaxPooling2D((2, 2)))
#Add a convnet for 3*3 and 128 of them, activation = relu
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
 #Flatten our convnet
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.summary()
#-------------------------------------
#GET DATA AND REFORMAT
#-------------------------------------
from keras.datasets import mnist
from keras.datasets import fashion_mnist
from keras.datasets import cifar10
from sklearn.model_selection import KFold
from tensorflow.keras.utils import to_categorical


if default == 'mnist' or default == 'mnist_fashion':
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = train_images.reshape((60000, 28, 28, 1))
    test_images = test_images.reshape((10000, 28, 28, 1))
#NORMALIZE
    train_images = train_images.astype('float32') / 255 
    test_images = test_images.astype('float32') / 255  
#DEBUGGING
    NKEEP=10000
    batch_size=int(0.05*NKEEP)
    print("batch_size",batch_size)
    rand_indices = np.random.permutation(train_images.shape[0])
    train_images=train_images[rand_indices[0:NKEEP],:,:]
    train_labels=train_labels[rand_indices[0:NKEEP]]
# exit()
#CONVERTS A CLASS VECTOR (INTEGERS) TO BINARY CLASS MATRIX.
    tmp=train_labels[0]
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)
    print(tmp, '-->',train_labels[0])
    print("train_labels shape:", train_labels.shape)


elif default == 'cifar10':
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
    train_images = train_images.astype('float32') / 255 
    test_images = test_images.astype('float32') / 255  
    NKEEP=10000
    batch_size=int(0.05*NKEEP)
    print("batch_size",batch_size)
    rand_indices = np.random.permutation(train_images.shape[0])
    train_images=train_images[rand_indices[0:NKEEP],:,:]
    train_labels=train_labels[rand_indices[0:NKEEP]]

#CONVERTS A CLASS VECTOR (INTEGERS) TO BINARY CLASS MATRIX.
    tmp=train_labels[0]
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)
    print(tmp, '-->',train_labels[0])
    print("train_labels shape:", train_labels.shape)

def visualization(df,sample):
    each = df[sample]
    plt.figure(figsize=(2,2))
    plt.axis('off')
    plt.imshow(each,interpolation = 'nearest')

dimension = [train_images.shape[0],train_images.shape[1],train_images.shape[2],train_images.shape[3]]


#-------------------------------------
#BUILD MODEL CNN
#-------------------------------------

if CNN == True:
    kfold = KFold(n_splits = folds, shuffle = True)
    fold_count = 1 
    acc_per_fold = []
    loss_per_fold = [] 
    for train, test in kfold.split(train_images,train_labels):
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(dimension[1], dimension[2], dimension[3])))
        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Conv2D(64, (3, 3), activation='relu')) 
        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Conv2D(128,(3,3),activation = "relu"))
        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Flatten())
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dense(10, activation='softmax'))

        model.compile(
            optimizer=optimize,
                    loss=Loss,
                    metrics=Metric)

        history = model.fit(train_images[train], train_labels[train], epochs = epochs, batch_size = batch_size)
        scores = model.evaluate(train_images[test], train_labels[test], batch_size = train_images[test].shape[0]) 
        print(f'Score for fold {fold_count}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
        acc_per_fold.append(scores[1] * 100)
        loss_per_fold.append(scores[0])
        fold_count += 1 
    print("average accuracy = ", np.mean(acc_per_fold))
    print("average loss = ", np.mean(loss_per_fold))
if DFF_ANN == True:
    train_images = train_images.reshape((dimension[0], dimension[1] * dimension[2] * dimension[3]))
    test_images = test_images.reshape((test_images.shape[0], test_images.shape[1] * test_images.shape[2] * test_images.shape[3]))
    kfold = KFold(n_splits=folds, shuffle=True)
    fold_count = 1
    acc_per_fold = []
    loss_per_fold = []
    for train, test in kfold.split(train_images, train_labels):
        model = models.Sequential()
        #ADD LAYERS
        model.add(layers.Dense(512, activation='relu', input_shape=(dimension[1] * dimension[2] * dimension[3],)))
        #SOFTMAX  --> 10 probability scores (summing to 1
        model.add(layers.Dense(10,  activation='softmax'))

        #COMPILATION (i.e. choose optimizer, loss, and metrics to monitor)
        model.compile(optimizer=optimize,
                    loss=Loss,
                    metrics=Metric)
        history = model.fit(train_images[train], train_labels[train], epochs = epochs, batch_size = batch_size)
        scores = model.evaluate(train_images[test], train_labels[test], batch_size = train_images[test].shape[0])
        print(f'Score for fold {fold_count}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
        acc_per_fold.append(scores[1] * 100)
        loss_per_fold.append(scores[0])
        fold_count += 1
    print("average accuracy = ", np.mean(acc_per_fold))
    print("average loss = ", np.mean(loss_per_fold))

#------------------Plot our accuracy and loss ----------------
plt.plot(history.history['accuracy'])
plt.plot(history.history['loss'])
plt.title('model results training history')
plt.ylabel('accuracy/loss')
plt.xlabel('epoch')
plt.legend(['Acuuracy', 'Loss'], loc='upper left')
plt.show()


#------------------Save our model------------------------------
if model_save :
    model.save("ourmodel")


#------------------Load our model------------------------------
def load_model(path):
    model = models.load_model(model_name)
    return model
if model_load: 
    model = load_model(model_name)


#-------------------------------------
#EVALUATE ON TEST DATA
#-------------------------------------
train_loss, train_acc = model.evaluate(train_images, train_labels, batch_size=batch_size)
test_loss, test_acc = model.evaluate(test_images, test_labels, batch_size=test_images.shape[0])
print('train_acc:', train_acc)
print('test_acc:', test_acc)


