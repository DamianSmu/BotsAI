import keras
import keras.optimizers
from keras import layers
from keras.optimizer_v2.adam import Adam
from tensorflow import keras
import tensorflow as tf


class NN:
    def __init__(self, map_size, local_size, input_depth, actions_number, lr):
        input_global = keras.Input(shape=(2 * map_size - 1, 2 * map_size - 1, input_depth), name='Game_state_global')
        input_local = keras.Input(shape=(local_size, local_size, input_depth), name='Game_state_local')

        layer_global = layers.Conv2D(32, 4, strides=2, activation="leaky_relu")(input_global)
        layer_global = layers.Conv2D(64, 4, strides=2, activation="leaky_relu")(layer_global)
        layer_global = layers.Conv2D(64, 4, strides=2, activation="leaky_relu")(layer_global)

        layer_global = layers.Flatten()(layer_global)
        layer_local = layers.Flatten()(input_local)

        f = layers.concatenate([layer_local, layer_global])
        f = layers.Dense(512, activation="leaky_relu")(f)

        output = layers.Dense(actions_number, activation="linear", name="Units_actions")(f)

        self.model = keras.Model(inputs=[input_global, input_local], outputs=output)
        self.model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=lr))
        print(self.model.summary())
        tf.keras.utils.plot_model(self.model, to_file="n2n", show_shapes=True, dpi=200)
