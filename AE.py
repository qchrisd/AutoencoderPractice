# -*- coding: utf-8 -*-
"""

Practice autoencoding with mnist

"""


#%% Import packages

from tensorflow import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


#%% Import mnist data

# Get mnist
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path="mnist.npz")


# normalize mnist
x_train = x_train/255
x_test = x_test/255

# Reshape for convolution layers
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

#%% Visualize mnist data

plt.imshow(x_train[1], cmap='gray')

#%% Build autoencoder

def autoencoder():
    
    ## Input
    input_image = keras.Input(shape=(28,28,1))
    
    ## Encoder
    enc_conv1 = keras.layers.Conv2D(32, 3, activation='relu', padding='same')(input_image)  # Convolution
    enc_pool1 = keras.layers.MaxPooling2D(2)(enc_conv1)  # Pooling to 14x14
    enc_conv2 = keras.layers.Conv2D(64, 3, activation='relu', padding='same')(enc_pool1)
    enc_pool2 = keras.layers.MaxPool2D(2)(enc_conv2) # Pooling to 7x7
    enc_conv3 = keras.layers.Conv2D(128, 3, activation='relu', padding='same', name='code')(enc_pool2)
    
    ## Decoder
    dec_conv1 = keras.layers.Conv2D(128, 3, activation='relu', padding='same')(enc_conv3) # Convolution
    dec_up1 = keras.layers.UpSampling2D(2)(dec_conv1)  # Up sample to 14x14
    dec_conv2 = keras.layers.Conv2D(64, 3, activation='relu', padding='same')(dec_up1)
    dec_up2 = keras.layers.UpSampling2D(2)(dec_conv2)  # up to 28x28
    dec_conv3 = keras.layers.Conv2D(32, 3, activation='relu', padding='same')(dec_up2)
    
    ## Output
    output = keras.layers.Conv2D(1, 3, activation='sigmoid', padding='same')(dec_conv3)
    
    ## Compile
    model = keras.Model(input_image, output)
    model.compile(optimizer='adam',
                  loss='mse',
                  metrics=['accuracy'])
    
    ## Return model
    return(model)


# This version is a simpler version with less compression of the image.
# This one runs better on my laptop and also seems to perform better.
# Perhaps 7x7 was too much compression of the image
def autoencoder_simple():
    
    ## Input
    input_image = keras.Input(shape=(28,28,1))
    
    ## Encoder
    enc_conv1 = keras.layers.Conv2D(8, 3, activation='relu', padding='same')(input_image)  # Convolution
    enc_pool1 = keras.layers.MaxPooling2D(2)(enc_conv1)  # Pooling to 14x14
    enc_conv2 = keras.layers.Conv2D(8, 3, activation='relu', padding='same', name='code')(enc_pool1)
    
    ## Decoder
    dec_conv1 = keras.layers.Conv2D(8, 3, activation='relu', padding='same')(enc_conv2) # Convolution
    dec_up1 = keras.layers.UpSampling2D(2)(dec_conv1)  # Up sample to 14x14
    dec_conv2 = keras.layers.Conv2D(8, 3, activation='relu', padding='same')(dec_up1)
    
    ## Output
    output = keras.layers.Conv2D(1, 3, activation='sigmoid', padding='same')(dec_conv2)
    
    ## Compile
    model = keras.Model(input_image, output)
    model.compile(optimizer='adam',
                  loss='mse',
                  metrics=['accuracy'])
    
    ## Return model
    return(model)

# Build the model from the function
ae = autoencoder_simple()

#%% Fit the model

# Create early stopping
stopping = keras.callbacks.EarlyStopping('val_loss', patience=2)

# Fit
ae_history = ae.fit(x_train, x_train,
                    batch_size=128,
                    epochs=5,
                    validation_split=.2,
                    callbacks=[stopping])

#%% Show a sample of predicted images

# Generates random indeces from the test set to plot
rand_nums = np.random.randint(0, 10000, 4)

# Make image reconstruction predictions
x_preds = ae.predict(x_test[rand_nums])

# Init the figure
fig = plt.figure(figsize=(20,15))

# first figure
fig.add_subplot(2, 4, 1)
plt.imshow(x_test[rand_nums[0]], cmap='gray')
fig.add_subplot(2, 4, 5)
plt.imshow(x_preds[0], cmap='gray')

# Second figure
fig.add_subplot(2, 4, 2)
plt.imshow(x_test[rand_nums[1]], cmap='gray')
fig.add_subplot(2, 4, 6)
plt.imshow(x_preds[1], cmap='gray')

# Third figure
fig.add_subplot(2, 4, 3)
plt.imshow(x_test[rand_nums[2]], cmap='gray')
fig.add_subplot(2, 4, 7)
plt.imshow(x_preds[2], cmap='gray')

# Second figure
fig.add_subplot(2, 4, 4)
plt.imshow(x_test[rand_nums[3]], cmap='gray')
fig.add_subplot(2, 4, 8)
plt.imshow(x_preds[3], cmap='gray')
    
#%% Transfer learning with the code

# Creates a new model with the encoder portion of the trained autoencoder
# We also need to freeze the model so that it isn't adjusting as we fit the new model
encoder = keras.Model(inputs=ae.input, outputs=ae.get_layer('code').output)
encoder.trainable = False

# A classifier network on top of the code
def classifier():
    
    ## Input layer for base image
    input_code = keras.Input(shape=(28,28))
    
    ## Get code from encoder
    # We need to set it to training=False to prevent any batch normalization layers
    # in the encoder from discarding important variance and mean information learned
    # when training the encoder.
    code = encoder(input_code, training=False)
    
    ## Classifier
    flat = keras.layers.Flatten()(code)
    dense1 = keras.layers.Dense(50, activation='relu')(flat)
    output = keras.layers.Dense(10, activation='softmax')(dense1)
    
    ## Compile model
    model = keras.Model(inputs=input_code, outputs=output)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    ## Return the model
    return(model)

c = classifier()

#%% Fit classifier

c_history = c.fit(x_train, y_train,
                  epochs=20,
                  batch_size=128,
                  callbacks=[stopping],
                  validation_split=.2)
    

#%% Evaluate the model

c.evaluate(x_test, y_test)
