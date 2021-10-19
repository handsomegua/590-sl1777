import os, shutil
from keras import layers
from keras import models
from keras import optimizers
from keras.applications.vgg16 import VGG16
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
original_dataset_dir = "/Users/hayashishishin/Downloads/dogs-vs-cats/train"

base_dir = os.path.join(os.path.dirname(os.getcwd()),'cats_and_dogs_samll')
if not os.path.exists(base_dir):
    os.mkdir(base_dir)
train_dir = os.path.join(base_dir,'train')
if not os.path.exists(train_dir):
    os.mkdir(train_dir)
validation_dir = os.path.join(base_dir,'validation')
if not os.path.exists(validation_dir):
    os.mkdir(validation_dir)
test_dir = os.path.join(base_dir,'test')
if not os.path.exists(test_dir):
    os.mkdir(test_dir)
train_cats_dir = os.path.join(train_dir, 'cats')
if not os.path.exists(train_cats_dir):
    os.mkdir(train_cats_dir)
train_dogs_dir = os.path.join(train_dir, 'dogs')
if not os.path.exists(train_dogs_dir):
    os.mkdir(train_dogs_dir)
validation_cats_dir = os.path.join(validation_dir, 'cats')
if not os.path.exists(validation_cats_dir):
    os.mkdir(validation_cats_dir)
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
if not os.path.exists(validation_dogs_dir):
    os.mkdir(validation_dogs_dir)
test_cats_dir = os.path.join(test_dir, 'cats')
if not os.path.exists(test_cats_dir):
    os.mkdir(test_cats_dir)
test_dogs_dir = os.path.join(test_dir, 'dogs')
if not os.path.exists(test_dogs_dir):
    os.mkdir(test_dogs_dir)
fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_cats_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_cats_dir, fname)
    shutil.copyfile(src, dst)


fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_cats_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_dogs_dir, fname)
    shutil.copyfile(src, dst)


fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_dogs_dir, fname)
    shutil.copyfile(src, dst)


fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_dogs_dir, fname)
    shutil.copyfile(src, dst)


from keras import layers
from keras import models
from keras import optimizers


# Create model building function 
def model_building():
    #initialize the model
    model = models.Sequential()
    #Add a  32 convnet for 3*3 , activation = relu, input image shape 150*150 * 3 color
    model.add(layers.Conv2D(32,(3,3),activation = 'relu',input_shape = (150,150,3)))
    #Add  max_pooling layer
    model.add(layers.MaxPooling2D((2,2)))
    #Add a convnet for 3*3 and 64 of them, activation = relu
    model.add(layers.Conv2D(64,(3,3),activation = 'relu'))
    #Add a max_pooling layer
    model.add(layers.MaxPooling2D((2,2)))
    #Add a convnet for 3*3 and 128 of them, activation = relu
    model.add(layers.Conv2D(128,(3,3),activation = 'relu'))
    #Add a max_pooling layer
    model.add(layers.MaxPooling2D((2,2)))
    #Add a convnet for 3*3 and 128 of them, activation = relu
    model.add(layers.Conv2D(128,(3,3),activation = 'relu'))
    #Add a max_pooling layer
    model.add(layers.MaxPooling2D((2,2)))
    #Flatten our convnet
    model.add(layers.Flatten())
    #Add a DFF network of 512 neuron
    model.add(layers.Dense(512, activation='relu'))
    #Output layer with 1 output node and sigmoid activation
    model.add(layers.Dense(1, activation='sigmoid'))
    #Show the summary of our CNN network
    model.summary()
    #Compile the model
    model.compile(loss='binary_crossentropy',
                optimizer='rmsprop',
                metrics=['acc'])
    return model
model = model_building()

#---------------Preprocess Data-------------------

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image


# define the generator function 
def generator():
    #Rescale image size
    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)
    #define a generator for train and validation datasets
    train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size = (150,150),
                                                    batch_size = 32,
                                                    class_mode = 'binary')
    validation_generator = test_datagen.flow_from_directory(validation_dir,
                                                    target_size = (150,150),
                                                    batch_size = 32,
                                                    class_mode = 'binary')
    return train_generator, validation_generator

train_generator, validation_generator = generator()

#Define a fit function
def fit(): 
    #fit function
    history = model.fit_generator(train_generator,
                                  steps_per_epoch=100,
                                  epochs=30,
                                  validation_data=validation_generator,
                                  validation_steps=50)
    return history
history = fit()
model.save('cats_and_dogs_small_1.h5')


#Define a cnn network with dropout


#----------------visualize parameters, uncomment if needed-------------------
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

# Visualization 
img_path =  "/Users/hayashishishin/Downloads/dogs-vs-cats" + '/train/cat.1.jpg'
img = image.load_img(img_path,target_size = (150,150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor,axis = 0)
img_tensor /= 255 


layer_outputs = [layer.output for layers in model.layers[:4]]
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(img_tensor)
first_layer_activation = activations[0]
print(first_layer_activation.shape)
plt.matshow(first_layer_activation[0, :, :, 4], cmap='viridis')
layer_names = []
for layer in model.layers[:4]:
    layer_names.append(layer.name)

images_per_row = 8
for layer_name, layer_activation in zip(layer_names, activations):
    #number of features in the feature map
    n_features = layer_activation.shape[-1]
    # The feature map has shape (1,size,size,number of features)
    size = layer_activation.shape[1]
    # tiles the activation channels in this matrix 
    n_cols = n_features // images_per_row
    display_grid = np.zeros((size * n_cols, images_per_row * size))
    #Tiles each filter into a big horizontal grid 
    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[0, :, :, col * images_per_row+row]
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col*size: (col + 1) * size,
                         row*size: (row + 1)*size] = channel_image
    scale = 1. / size
    plt.figure(figsize=(scale*display_grid.shape[1], scale*display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
    plt.show()