import time

import numpy as np
import numpy.random
from keras import Sequential
from keras.src.layers import Dense
from keras.src.optimizers import Adam

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

input_space_size = 5
output_space_size = 2
model = Sequential([
			Dense(128, activation='relu', kernel_initializer='random_normal', input_shape=(input_space_size,)),
			Dense(128, activation='relu', kernel_initializer='random_normal', input_shape=(input_space_size,)),
			Dense(64, activation='relu', kernel_initializer='random_normal', input_shape=(input_space_size,)),
			Dense(32, activation='relu', kernel_initializer='random_normal', input_shape=(input_space_size,)),
			Dense(output_space_size, activation='linear', kernel_initializer='random_normal')
		])
model.compile(loss="mse", optimizer=Adam(learning_rate=0.001))

data_batch_size = 20
data: np.ndarray = np.array(np.random.uniform(0., 100., (data_batch_size, 1, input_space_size)))
predictions = []

print("> Mode #1 : model.predict(datum)... ")
predictions.append([])
for data_row in data:
	predictions[-1].append(model.predict(data_row, verbose=0))

print("> Mode #2 : model(datum)... ", end='')
predictions.append([])
for data_row in data:
	predictions[-1].append(model(data_row))

print("> Mode #3 : model.predict_batch(data)... ")

d = np.reshape(data, (data_batch_size, input_space_size))
predictions.append(model.predict_on_batch(d))

print("> Mode #4 : model(data)... ")
d = np.reshape(data, (data_batch_size, input_space_size))
predictions.append(model(d))  # this is definitely the best one

for p in predictions:
	print(p)

