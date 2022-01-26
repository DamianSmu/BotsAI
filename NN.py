import keras
import keras.optimizers
from keras import layers
from keras.optimizer_v2.adam import Adam
from tensorflow import keras
import tensorflow as tf


class NN:
    def __init__(self, map_size, local_size, input_depth, actions, lr):
        shape = (2 * map_size - 1, 2 * map_size - 1, input_depth)
        in_global = keras.Input(shape)
        in_local = keras.Input(shape=(local_size, local_size, input_depth))

        l_global = layers.Conv2D(32, 4, strides=2, activation="leaky_relu")(in_global)
        l_global = layers.Conv2D(64, 4, strides=2, activation="leaky_relu")(l_global)
        l_global = layers.Conv2D(64, 4, strides=2, activation="leaky_relu")(l_global)

        l_global = layers.Flatten()(l_global)
        l_local = layers.Flatten()(in_local)

        f = layers.concatenate([l_local, l_global])
        f = layers.Dense(512, activation="leaky_relu")(f)

        out = layers.Dense(actions, activation="linear")(f)

        self.model = keras.Model(inputs=[in_global, in_local], outputs=out)
        self.model.compile(loss='mse', optimizer=Adam(learning_rate=lr))
        print(self.model.summary())
        tf.keras.utils.plot_model(self.model, to_file="n2n", show_shapes=True, dpi=200)
