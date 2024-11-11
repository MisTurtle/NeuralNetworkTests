import time

import numpy as np
import numpy.random
from keras import Sequential
from keras.src.layers import Dense
from keras.src.optimizers import Adam

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def init_timer() -> float:
	global timer
	ret = time.time() - timer
	timer = time.time()
	return ret


def new_data():
	global data
	data = np.array(np.random.uniform(0., 100., (data_sizes[current_data_size], 1, input_space_size)))
	print("Initialized random data of size " + str(data.shape))


data_sizes = [10, 100, 500, 5000]
current_data_size = 0
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
timer = time.time()
ys = [[], [], [], []]

data: np.ndarray


for i in range(len(data_sizes)):
	print("\n\n-------------------\nRunning for data size = %d\n-------------------\n\n" % data_sizes[i])
	current_data_size = i
	new_data()

	# print("> Benchmark #1 : model.predict(datum), 1 by 1... ", end='')
	# init_timer()
	# for data_row in data:
	# 	model.predict(data_row, verbose=0)
	# dt = init_timer()
	# ys[0].append(dt)
	# print("Done in %.4fs" % dt)

	print("> Benchmark #2 : model(datum), 1 by 1... ", end='')
	init_timer()
	for data_row in data:
		model(data_row)
	dt = init_timer()
	ys[1].append(dt)
	print("Done in %.4fs" % dt)

	print("> Benchmark #3 : model.predict_batch(data)... ", end='')
	init_timer()
	d = np.reshape(data, (data_sizes[current_data_size], input_space_size))
	model.predict_on_batch(d)
	dt = init_timer()
	ys[2].append(dt)
	print("Done in %.4fs" % dt)

	print("> Benchmark #4 : model(data)... ", end='')
	init_timer()
	d = np.reshape(data, (data_sizes[current_data_size], input_space_size))
	model(d)
	dt = init_timer()
	ys[3].append(dt)
	print("Done in %.4fs" % dt)


plt.xlabel("Processing Time (s)")
plt.ylabel("Data Size (rows)")

for y in ys:
	print(y, data_sizes)
	if len(y) != len(data_sizes):
		continue
	plt.plot(data_sizes, y)
plt.show()

