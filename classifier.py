# import numpy as np
# import keras
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
# from keras.optimizers import SGD
# from keras.layers.advanced_activations import LeakyReLU
from gendataset import Gendataset
from spectrogram import Spectrogram
import os

train_vocals_dir = ""
train_no_vocals_dir = ""
test_vocals_dir = "C:/Users/tuanh/PycharmProjects/CNN Vocal Classifier/test/Vocal Versions"
test_no_vocals_dir = "C:/Users/tuanh/PycharmProjects/CNN Vocal Classifier/test/Instrumental Versions"

# Generate full train set
# gendata_train_vocals = Gendataset(train_vocals_dir, True)
# X_train_vocals = gendata_train_vocals.X
# Y_train_vocals = gendata_train_vocals.Y
# gendata_train_no_vocals = Gendataset(train_no_vocals_dir, False)
# X_train_no_vocals = gendata_train_no_vocals.X
# Y_train_no_vocals = gendata_train_no_vocals.Y

# Generate full test set
# gendata_test_vocals = Gendataset(test_vocals_dir, True)
# X_test_vocals = gendata_test_vocals.X
# Y_test_vocals = gendata_test_vocals.Y
# gendata_test_no_vocals = Gendataset(test_no_vocals_dir, False)
# X_test_no_vocals = gendata_test_no_vocals.X
# Y_test_no_vocals = gendata_test_no_vocals.Y
# print(X_test_no_vocals.shape)
# print(X_test_vocals.shape)

spec = Spectrogram("Al James - Schoolboy Facination.stem.wav", True)
x = spec.get_x()

# model = Sequential()
# model.add(Conv2D(16, (3, 3), padding='same', input_shape=(513, 25, 1)))
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
# model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50)
