import keras
import keras.optimizers
import numpy as np
from keras import layers
from keras.models import load_model
from keras.optimizer_v2.adam import Adam
from tensorflow import keras
import tensorflow as tf
from tensorflow_addons.layers import NoisyDense


class NN:

    def __init__(self, map_size, local_size, depth, actions_number, lr):
        input_global = keras.Input(shape=(2 * map_size - 1, 2 * map_size - 1, depth), name='Game_state_global')
        input_local = keras.Input(shape=(local_size, local_size, depth), name='Game_state_local')

        layer_global = layers.Conv2D(32, 4, strides=2, activation="leaky_relu")(input_global)
        layer_global = layers.Conv2D(64, 4, strides=2, activation="leaky_relu")(layer_global)
        layer_global = layers.Conv2D(64, 4, strides=2, activation="leaky_relu")(layer_global)

        # layer_global = layers.MaxPool2D((2, 2))(layer_global)
        # layer_global = layers.Conv2D(32, 3, strides=1, activation="relu")(layer_global)
        # layer_global_2 = layers.Conv2D(32, 3, strides=1, activation="relu")(layer_global_1)

        layer_global = layers.Flatten()(layer_global)
        layer_local = layers.Flatten()(input_local)

        f = layers.concatenate([layer_local, layer_global])
        f = layers.Dense(512, activation="leaky_relu")(f)

        # x = layers.Dense(actions_number + 1, activation='linear')(f)
        # output = layers.Lambda(
        #     lambda i: keras.backend.expand_dims(i[:, 0], -1) + i[:, 1:] - keras.backend.mean(i[:, 1:], keepdims=True),
        #     output_shape=(actions_number,))(x)

        output = layers.Dense(actions_number, activation="linear", name="Units_actions")(f)

        self.model = keras.Model(inputs=[input_global, input_local], outputs=output)

        self.model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=lr))
        print(self.model.summary())

    def load_model(self, filepath):
        self.model = load_model(filepath)
        return self.model
