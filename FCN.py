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


# ### FCN

# In[11]:


# location of VGG weights
VGG_Weights_path = "'/home/kiara/segmentationUNET/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"


# In[12]:


c1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
c1 = tf.keras.layers.Dropout(0.5)(c1)
c1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

c2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
c2 = tf.keras.layers.Dropout(0.5)(c2)
c2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
 
c3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
c3 = tf.keras.layers.Dropout(0.5)(c3)
c3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
 
c4 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
c4 = tf.keras.layers.Dropout(0.5)(c4)
c4 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)
 
c5 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
c5 = tf.keras.layers.Dropout(0.5)(c5)
c5 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
c5 = tf.keras.layers.Dropout(0.5)(c5)

u6 = tf.keras.layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c5)
c6 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
c6 = tf.keras.layers.Dropout(0.5)(c6)
c6 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
 
u7 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c6)
c7 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
c7 = tf.keras.layers.Dropout(0.5)(c7)
c7 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
 
u8 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c7)
u8 = tf.keras.layers.concatenate([u8, c2])
c8 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
c8 = tf.keras.layers.Dropout(0.5)(c8)
c8 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
 
u9 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c8)
u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
c9 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
c9 = tf.keras.layers.Dropout(0.5)(c9)
c9 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

# In[13]:


outputs = tf.keras.layers.Conv2D(3, (1,1), activation='sigmoid')(c9)


# ### -----------------------------------------------------------------------------------------------------------------------------------------------------------

# In[14]:


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


# In[15]:


def iou_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
    union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou


# In[16]:


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


# In[17]:


from tensorflow import keras
from tensorflow.keras import optimizers
model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
opt = keras.optimizers.Adam(learning_rate=0.00001)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=[dice_coef, precision, recall, iou_coef, 'acc'])
model.summary()


# ## Image Manipulation

# 

# In[18]:


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
checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath='/home/kiara/testing_SEGMENTATION/fcn.h5', verbose = 2, save_weights_only = True)
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience = 20, monitor = 'val_loss'),
    tf.keras.callbacks.TensorBoard(log_dir = 'fcn')
]


# FIT MODEL
results = model.fit(X_train, Y_train, validation_split = 0.1, batch_size = 4, epochs = 200, callbacks=callbacks)


# In[28]:


model.save('model_fcn')


# In[29]:


get_ipython().system('tensorboard --logdir=fcn/ --host localhost --port 8000')


# In[30]:


model.evaluate(X_train, Y_train, verbose=1)


# In[31]:


model.evaluate(X_test, Y_test, verbose=1)

