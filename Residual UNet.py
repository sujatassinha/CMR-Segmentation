#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing the necessary libraries
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random
from tqdm import tqdm
import pandas as pd
from skimage.io import imread, imshow
from skimage.transform import resize
import scipy.ndimage
import IPython
from sklearn.preprocessing import StandardScaler
import sklearn.model_selection
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# defining the training image and label(mask) directories
base_dir = '/home/kiara/testing_SEGMENTATION/'
train_img_dir = os.path.join(base_dir, 'SCCOR_Images/ShortAxis/')
train_label_dir = os.path.join(base_dir, 'SCCOR_labels/ShortAxis')


# In[3]:


# obtaining the training image (and corresponding label (masks)) file names as a list
train_img_fname = os.listdir(train_img_dir)
train_label_fname = train_img_fname


# In[4]:


# shuffling the image list randomply and saving it
train_img_fnames = random.sample(train_img_fname, len(train_img_fname))
train_label_fnames = train_img_fnames


# In[5]:


training_dataset, test_dataset = sklearn.model_selection.train_test_split(train_img_fnames, test_size=0.1)


# ## Test images

# In[6]:


# test on different dataset
test_img_dir = '/home/kiara/testing_SEGMENTATION/SCCOR_Images/ShortAxis/'
test_label_dir = '/home/kiara/testing_SEGMENTATION/SCCOR_labels/ShortAxis/'
test_img_fnames = os.listdir(test_img_dir)
test_label_fnames = test_img_fnames
print(test_img_fnames[:10])
print("\n")
print(test_label_fnames[:10])


# In[7]:


test_img_fnames = test_img_fnames
test_label_fnames = test_img_fnames


# In[8]:


train_img_fnames = training_dataset
train_label_fnames = training_dataset


# ## Model

# In[9]:


IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNEL = 1
IMG_CHANNELS = 3


# In[10]:


# defining input layer
inputs = tf.keras.layers.Input((IMG_WIDTH, IMG_HEIGHT, IMG_CHANNEL))
# pixels to floating point numbers
s = tf.keras.layers.Lambda(lambda x: (x/255))(inputs)
print(s)


# ### UNET with drop-out

# In[11]:


# taken directly from the original implementation https://arxiv.org/pdf/1711.10684.pdf
def bn_act(x, act=True):
    x = tf.keras.layers.BatchNormalization()(x)
    if act == True:
        x = tf.keras.layers.Activation("relu")(x)
    return x

def conv_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    conv = bn_act(x)
    conv = tf.keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(conv)
    return conv

def stem(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    conv = tf.keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
    conv = conv_block(conv, filters, kernel_size=kernel_size, padding=padding, strides=strides)
    
    shortcut = tf.keras.layers.Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
    shortcut = bn_act(shortcut, act=False)
    
    output = tf.keras.layers.Add()([conv, shortcut])
    return output

def residual_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    res = conv_block(x, filters, kernel_size=kernel_size, padding=padding, strides=strides)
    res = conv_block(res, filters, kernel_size=kernel_size, padding=padding, strides=1)
    
    shortcut = tf.keras.layers.Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
    shortcut = bn_act(shortcut, act=False)
    
    output = tf.keras.layers.Add()([shortcut, res])
    return output

def upsample_concat_block(x, xskip):
    u = tf.keras.layers.UpSampling2D((2, 2))(x)
    c = tf.keras.layers.concatenate([u, xskip])
    return c


# In[12]:

# taken directly from the original implementation https://arxiv.org/pdf/1711.10684.pdf
def ResUNet():
    f = [16, 32, 64, 128, 256]
    inputs = keras.layers.Input((256, 256, 1))
    
    ## Encoder
    e0 = inputs
    e1 = stem(e0, f[0])
    e2 = residual_block(e1, f[1], strides=2)
    e3 = residual_block(e2, f[2], strides=2)
    e4 = residual_block(e3, f[3], strides=2)
    e5 = residual_block(e4, f[4], strides=2)
    
    ## Bridge
    b0 = conv_block(e5, f[4], strides=1)
    b1 = conv_block(b0, f[4], strides=1)
    
    ## Decoder
    u1 = upsample_concat_block(b1, e4)
    d1 = residual_block(u1, f[4])
    
    u2 = upsample_concat_block(d1, e3)
    d2 = residual_block(u2, f[3])
    
    u3 = upsample_concat_block(d2, e2)
    d3 = residual_block(u3, f[2])
    
    u4 = upsample_concat_block(d3, e1)
    d4 = residual_block(u4, f[1])
    
    outputs = keras.layers.Conv2D(3, (1, 1), padding="same", activation="sigmoid")(d4)
    model = keras.models.Model(inputs, outputs)
    return model


# ### -----------------------------------------------------------------------------------------------------------------------------------------------------------

# In[13]:


import tensorflow.keras.backend as K
def dice_coef(y_true, y_pred):
    """
    Function to calculate dice coefficient
    
    Parameters
    ----------
    y_true : numpy array of actual masks
    y_pred : numpy array of predicted masks
    
    Returns
    -------
    dice coefficient
    
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())


# In[14]:


def iou_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
    union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou


# In[15]:


def precision(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision
def recall(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


# In[16]:


# for ResUnet
from tensorflow import keras
from tensorflow.keras import optimizers
model = ResUNet()
opt = keras.optimizers.Adam(learning_rate=0.00001)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=[dice_coef, precision, recall, iou_coef, 'acc'])


# In[15]:


model.summary()


# ## Image Manipulation

# 

# In[17]:


# creating an array of the same dimension as the input images
X_train = np.zeros((len(train_img_fnames), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL), dtype = np.float32)
Y_train = np.zeros((len(train_img_fnames), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype = np.float32)


# ### X_train 

# In[19]:


## RUn THIS ONLY
print("Resizing train images")
from numpy import asarray
from PIL import Image
for n, id_ in tqdm(enumerate(train_img_fnames), total=len(train_img_fnames)):
    #n=n*4
    path = base_dir
    img = imread(path + 'SCCOR_Images/ShortAxis/' + id_) # read the image
    pixels=asarray(img).astype('float32')
    pixels = resize(pixels, (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL), mode = 'constant', preserve_range = True)
    # convert from integers to floats
    pixels = pixels.astype('float32')
    # normalize to the range 0-1
    pixels /= 255.0
    # confirm the normalization
    X_train[n] = pixels.astype('float32')
    
    
    # Remove comments to perform Augmentation
'''    print("-----------------------CLAHE and ROTATE------------------")
    img = cv2.imread((path + 'SCCOR_Images/ShortAxis/' + id_), IMG_CHANNEL) # read the image
    # rotate the image
    r_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    #Converting image to LAB Color so CLAHE can be applied to the luminance channel
    lab_img= cv2.cvtColor(r_img, cv2.COLOR_BGR2LAB)
    #Splitting the LAB image to L, A and B channels, respectively
    l, a, b = cv2.split(lab_img)
    #Apply histogram equalization to the L channel
    equ = cv2.equalizeHist(l)
    #Combine the Hist. equalized L-channel back with A and B channels
    updated_lab_img1 = cv2.merge((equ,a,b))
    #Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    clahe_img = clahe.apply(l)
    #Combine the CLAHE enhanced L-channel back with A and B channels
    updated_lab_img2 = cv2.merge((clahe_img,a,b))
    #Convert LAB image back to color (RGB)
    CLAHE_img = cv2.cvtColor(updated_lab_img2, cv2.COLOR_LAB2BGR)
    pixels=asarray(CLAHE_img).astype('float32')
    pixels = resize(pixels, (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL), mode = 'constant', preserve_range = True)
    # convert from integers to floats
    pixels = pixels.astype('float32')
    # normalize to the range 0-1
    pixels /= 255.0
    X_train[n+1] = pixels.astype('float32')

    print("-----------------------CLAHE ONLY ------------------")
    img = cv2.imread((path + 'SCCOR_Images/ShortAxis/' + id_), IMG_CHANNEL) # read the image
    # rotate the image
    r_img = img
    #Converting image to LAB Color so CLAHE can be applied to the luminance channel
    lab_img= cv2.cvtColor(r_img, cv2.COLOR_BGR2LAB)
    #Splitting the LAB image to L, A and B channels, respectively
    l, a, b = cv2.split(lab_img)
    #Apply histogram equalization to the L channel
    equ = cv2.equalizeHist(l)
    #Combine the Hist. equalized L-channel back with A and B channels
    updated_lab_img1 = cv2.merge((equ,a,b))
    #Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    clahe_img = clahe.apply(l)
    #Combine the CLAHE enhanced L-channel back with A and B channels
    updated_lab_img2 = cv2.merge((clahe_img,a,b))
    #Convert LAB image back to color (RGB)
    CLAHE_img = cv2.cvtColor(updated_lab_img2, cv2.COLOR_LAB2BGR)
    pixels=asarray(CLAHE_img).astype('float32')
    pixels = resize(pixels, (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL), mode = 'constant', preserve_range = True)
    # convert from integers to floats
    pixels = pixels.astype('float32')
    # normalize to the range 0-1
    pixels /= 255.0
    X_train[n+2] = pixels.astype('float32')
    
    print("-----------------------CLAHE AND ROTATE COUNTER ONLY------------------")
    img = cv2.imread((path + 'SCCOR_Images/ShortAxis/' + id_), 1) # read the image
    # rotate the image
    r_img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    #Converting image to LAB Color so CLAHE can be applied to the luminance channel
    lab_img= cv2.cvtColor(r_img, cv2.COLOR_BGR2LAB)
    #Splitting the LAB image to L, A and B channels, respectively
    l, a, b = cv2.split(lab_img)
    #Apply histogram equalization to the L channel
    equ = cv2.equalizeHist(l)
    #Combine the Hist. equalized L-channel back with A and B channels
    updated_lab_img1 = cv2.merge((equ,a,b))
    #Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    clahe_img = clahe.apply(l)
    #Combine the CLAHE enhanced L-channel back with A and B channels
    updated_lab_img2 = cv2.merge((clahe_img,a,b))
    #Convert LAB image back to color (RGB)
    CLAHE_img = cv2.cvtColor(updated_lab_img2, cv2.COLOR_LAB2BGR)
    pixels=asarray(CLAHE_img).astype('float32')
    pixels = resize(pixels, (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL), mode = 'constant', preserve_range = True)
    # convert from integers to floats
    pixels = pixels.astype('float32')
    # normalize to the range 0-1
    pixels /= 255.0
    X_train[n+3] = pixels.astype('float32')'''


# ### Y_train

# In[20]:


## RUn THIS ONLY
print("Resizing train images")
from numpy import asarray
from PIL import Image
for n, id_ in tqdm(enumerate(train_img_fnames), total=len(train_label_fnames)):
    #n=n*4
    path = base_dir
    img = imread(path + 'SCCOR_labels/ShortAxis/' + id_) # read the image
    pixels=asarray(img).astype('float32')
    pixels = resize(pixels, (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), mode = 'constant', preserve_range = True)
    # convert from integers to floats
    pixels = pixels.astype('float32')
    # normalize to the range 0-1
    pixels /= 255.0
    Y_train[n] = pixels.astype('float32')
    
'''    # clahe and rotate
    img = imread(path + 'SCCOR_labels/ShortAxis/' + id_) # read the image
    r_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    pixels1=asarray(r_img).astype('float32')
    pixels1 = resize(pixels1, (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), mode = 'constant', preserve_range = True)
    # convert from integers to floats
    pixels1 = pixels1.astype('float32')
    # normalize to the range 0-1
    pixels1 /= 255.0
    # confirm the normalization
    Y_train[n+1] = pixels1.astype('float32')
    
    # clahe only
    Y_train[n+2] = Y_train[n]
    
    # clahe and rotate counter
    img = imread(path + 'SCCOR_labels/ShortAxis/' + id_) # read the image
    r_img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    pixels1=asarray(r_img).astype('float32')
    pixels1 = resize(pixels1, (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), mode = 'constant', preserve_range = True)
    # convert from integers to floats
    pixels1 = pixels1.astype('float32')
    # normalize to the range 0-1
    pixels1 /= 255.0
    # confirm the normalization
    Y_train[n+3] = pixels1.astype('float32')'''


# In[21]:


# plotting an image
seed = 17
np.random.seed = seed
image_x = random.randint(0, len(train_img_fnames)) # generate a random number between 0 and length of training ids
imshow(np.squeeze(X_train[image_x]))
#plt.savefig("image.pdf", format='pdf')
plt.show()


# In[22]:


imshow(np.squeeze(Y_train[image_x]))
#plt.savefig("label.pdf", format='pdf')
plt.show()


# In[23]:


# test images
X_test = np.zeros((len(test_img_fnames), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL), dtype = np.float32)
sizes_test = []
print("Resizing test images")
for n, id_ in tqdm(enumerate(test_img_fnames), total=len(test_img_fnames)):
    path = base_dir
    img = imread(path + 'SCCOR_Images/ShortAxis/' + id_) # read the image
    # Uncomment to test on HELIX Dataset
    #img = imread('/media/kiara/My Passport/HELIX/image/' + id_)
    pixels=asarray(img).astype('float32')
    pixels = resize(pixels, (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL), mode = 'constant', preserve_range = True)
    
    # convert from integers to floats
    pixels = pixels.astype('float32')
    
    # normalize to the range 0-1
    pixels /= 255.0
    X_test[n] = pixels.astype('float32')


# In[24]:


Y_test = np.zeros((len(test_label_fnames), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype = np.float32)
print("Resizing test images")
from numpy import asarray
from PIL import Image
for n, id_ in tqdm(enumerate(test_img_fnames), total=len(test_img_fnames)):
    #path = base_dir
    img = imread(path + 'SCCOR_labels/ShortAxis/' + id_) # read the image
    #img = imread('/media/kiara/My Passport/HELIX/label/ShortAxis/' + id_)
    pixels=asarray(img).astype('float32')
    pixels = resize(pixels, (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), mode = 'constant', preserve_range = True)
    
    # convert from integers to floats
    pixels = pixels.astype('float32')
    
    # normalize to the range 0-1
    pixels /= 255.0
    Y_test[n] = pixels.astype('float32')


# In[25]:


# plotting an image
seed = 17
np.random.seed = seed
image_x = random.randint(0, len(test_img_fnames)) # generate a random number between 0 and length of training ids
imshow(np.squeeze(X_test[image_x]))
#plt.savefig("image.pdf", format='pdf')
plt.show()


# In[26]:


imshow(np.squeeze(Y_test[image_x]))
#plt.savefig("label.pdf", format='pdf')
plt.show()


# ## Model Training and Validation

# In[27]:


# model checkpoint
checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath='/home/kiara/testing_SEGMENTATION/resunet.h5', verbose = 2, save_weights_only = True)
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience = 20, monitor = 'val_loss'),
    tf.keras.callbacks.TensorBoard(log_dir = 'ResUNet')
]


# FIT MODEL
results = model.fit(X_train, Y_train, validation_split = 0.1, batch_size = 4, epochs = 200, callbacks=callbacks)


# In[28]:


model.save('model_ResUNet')


# In[29]:


get_ipython().system('tensorboard --logdir=ResUNet/ --host localhost --port 8000')


# In[30]:


model.evaluate(X_train, Y_train, verbose=1)


# In[31]:


model.evaluate(X_test, Y_test, verbose=1)

