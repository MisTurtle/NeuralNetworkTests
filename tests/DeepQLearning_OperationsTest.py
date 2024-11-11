import time
import tensorflow as tf
import numpy as np
import numpy.random
from keras import Sequential
from keras.src.layers import Dense
from keras.src.optimizers import Adam

input_space_size = 5
output_space_size = 2
model_1 = Sequential([
			Dense(128, activation='relu', kernel_initializer='random_normal', input_shape=(input_space_size,)),
			Dense(128, activation='relu', kernel_initializer='random_normal', input_shape=(input_space_size,)),
			Dense(64, activation='relu', kernel_initializer='random_normal', input_shape=(input_space_size,)),
			Dense(32, activation='relu', kernel_initializer='random_normal', input_shape=(input_space_size,)),
			Dense(output_space_size, activation='linear', kernel_initializer='random_normal')
		])
model_2 = Sequential([
			Dense(128, activation='relu', kernel_initializer='random_normal', input_shape=(input_space_size,)),
			Dense(128, activation='relu', kernel_initializer='random_normal', input_shape=(input_space_size,)),
			Dense(64, activation='relu', kernel_initializer='random_normal', input_shape=(input_space_size,)),
			Dense(32, activation='relu', kernel_initializer='random_normal', input_shape=(input_space_size,)),
			Dense(output_space_size, activation='linear', kernel_initializer='random_normal')
		])
model_1.compile(loss="mse", optimizer=Adam(learning_rate=0.001))
model_2.compile(loss="mse", optimizer=Adam(learning_rate=0.001))
data_batch_size = 5

states: np.ndarray = np.array(np.random.uniform(0., 100., (data_batch_size, input_space_size)))
next_states: np.ndarray = np.array(np.random.uniform(0., 100., (data_batch_size, input_space_size)))
rewards = np.array(np.random.uniform(0., 100., data_batch_size))
ends = np.array(np.random.randint(0, 2, data_batch_size))
actions = np.argmax(model_1(states), axis=1)

print("Generated States: ", states)
print("Generated Rewards: ", rewards)
print("Simulation Ends: ", ends)
print("Picked actions: ", actions)

# Predict states using the current model
state_predictions = model_2(states)
next_state_predictions = model_2(states)

print("Predictions: ", state_predictions)
print("Next Predictions: ", next_state_predictions)

# Compute target rewards depending on whether the episode ended
best_actions = tf.reduce_max(next_state_predictions, axis=1)
target_values = rewards + (1 - ends) * best_actions

# Create target tensors by replacing the picked action's output by the newly computed target
target_tensor = state_predictions.numpy()  # A copy is needed because tensors are readonly
target_tensor[np.arange(data_batch_size), actions] = target_values

print("Best Actions: ", best_actions)
print("Target prediction: ", target_tensor)

