
# coding: utf-8

# In[61]:


import numpy as np
# import keras
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
# from keras.optimizers import SGD
# from keras.layers.advanced_activations import LeakyReLU
from gendataset import Gendataset
from spectrogram import Spectrogram
import os

train_vocals_dir = "C:/Users/numbe/Documents/Academics/Coursework/CS 129 - Machine Learning/Final Project/musdb18/train/Vocal Versions"
train_no_vocals_dir = "C:/Users/numbe/Documents/Academics/Coursework/CS 129 - Machine Learning/Final Project/musdb18/train/Instrumental Versions"
test_vocals_dir = "C:/Users/numbe/Documents/Academics/Coursework/CS 129 - Machine Learning/Final Project/musdb18/test/Vocal Versions"
test_no_vocals_dir = "C:/Users/numbe/Documents/Academics/Coursework/CS 129 - Machine Learning/Final Project/musdb18/test/Instrumental Versions"


# Generate full train set

n_samples_train = 46200
X_train = np.zeros((2*n_samples_train, 513, 23))
Y_train = np.zeros((2*n_samples_train, 1))

gendata_train_vocals = Gendataset(train_vocals_dir, True, n_samples_train)
X_train[0:n_samples_train,:,:] = gendata_train_vocals.X
Y_train[0:n_samples_train,1] = gendata_train_vocals.Y

gendata_train_no_vocals = Gendataset(train_no_vocals_dir, False, n_samples_train)
X_train[n_samples_train:,:,:] = gendata_train_no_vocals.X
Y_train[n_samples_train:, 1] = gendata_train_no_vocals.Y 

# Can make this more efficient by just generating one large empty matrix ~[X, Y] with a total 
# number of rows twice that of X or Y and then shuffle it, and then separate out X_train and Y_train
# but first would have to make X unrolled...

# X_train_unshuffled = (np.concatenate((X_train_vocals, X_train_no_vocals), axis=0))
# Y_train_unshuffled = (np.concatenate((Y_train_vocals, Y_train_no_vocals), axis=0))
# Matrix_train = np.random.shuffle(np.concatenate((X_train_unshuffled, Y_train_unshuffled), axis=1))

# X_train = Matrix_train[:,:-1]
# print(X_train.shape)
# Y_train = Matrix_train[:,Matrix_train.shape[1]-1]
# print(Y_train.shape)

# # Generate full test set

n_samples_test = 24505
X_test = np.zeros((2*n_samples_test, 513, 23))
#Y_test = np.zeros((2*n_samples_test, 1))
Y_test = np.zeros((2*n_samples_test))

gendata_test_vocals = Gendataset(test_vocals_dir, True, n_samples_test)
X_test[0:n_samples_test,:,:] = gendata_test_vocals.X
#Y_test[0:n_samples_test,1] = gendata_test_vocals.Y
Y_test[0:n_samples_test] = gendata_test_vocals.Y

gendata_test_no_vocals = Gendataset(test_no_vocals_dir, False , n_samples_test)
X_test[n_samples_test:,:,:] = gendata_test_no_vocals.X
#Y_test[n_samples_test:,1] = gendata_test_no_vocals.Y
Y_test[n_samples_test:] = gendata_test_no_vocals.Y

print(X_test_no_vocals.shape)
print(X_test_vocals.shape)


#spec = Spectrogram("Al James - Schoolboy Facination.stem.wav", True)
#x = spec.get_x()

# model = Sequential()
# model.add(Conv2D(16, (3, 3), padding='same', input_shape=(513, 23, 1)))
# model.add(LeakyReLU())
# model.add(Conv2D(16, (3, 3), padding='same'))
# model.add(LeakyReLU())
# model.add(MaxPooling2D(pool_size=(3, 3)))
# model.add(Dropout(0.25))
# model.add(Conv2D(16, (3, 3), padding='same'))
# model.add(LeakyReLU())
# model.add(Conv2D(16, (3, 3), padding='same'))
# model.add(LeakyReLU())
# model.add(MaxPooling2D(pool_size=(3, 3)))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(64))
# model.add(LeakyReLU())
# model.add(Dropout(0.5))
# model.add(Dense(1, activation='sigmoid'))
# sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
# model.compile(loss=keras.losses.binary_crossentropy, optimizer=sgd, metrics=['accuracy'])
# model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=50)


# In[60]:


# del X_train
# del Y_train
# #del Matrix_train
del X_train_vocals 
del Y_train_vocals
del X_train_no_vocals 
del Y_train_no_vocals 


# In[ ]:


# X_train_unshuffled = (np.concatenate((X_train_vocals, X_train_no_vocals), axis=0))
# Y_train_unshuffled = (np.concatenate((Y_train_vocals, Y_train_no_vocals), axis=0))
# print(X_train_unshuffled.shape)
# print(Y_train_unshuffled.shape)
# Matrix_train = np.random.shuffle(np.concatenate((X_train_unshuffled, Y_train_unshuffled), axis=1))

# X_train = Matrix_train[:,:-1]
# print(X_train.shape)
# Y_train = Matrix_train[:,Matrix_train.shape[1]-1]
# print(Y_train.shape)


# In[ ]:


# rng_state = np.random.get_state()
# np.random.shuffle(X_train_unshuffled)
# np.random.set_state(rng_state)
# np.random.shuffle(Y_train_unshuffled)

# print(X_train_unshuffled.shape)
# print(Y_train_unshuffled.shape)


# In[ ]:


Y_train_unshuffled[0:10]
X_train_unshuffled[0:10,:,:]


# In[59]:


X_train = np.zeros((2*n_samples_train, 513, 23))
Y_train = np.zeros((2*n_samples_train)) #Change to Y_train = np.zeros((2*n_samples_train,1)) 

X_train[0:n_samples_train,:,:] = gendata_train_vocals.X
Y_train[0:n_samples_train] = gendata_train_vocals.Y #Change to Y_train[0:n_samples_train,1] = gendata_train_vocals.Y
X_train[n_samples_train:,:,:] = gendata_train_no_vocals.X
Y_train[n_samples_train:] = gendata_train_no_vocals.Y #Change to Y_train[n_samples_train:,1] = gendata_train_no_vocals.Y

