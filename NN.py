import keras
import keras.optimizers
from keras import layers
from keras.models import load_model
from keras.optimizer_v2.adam import Adam
from tensorflow import keras


class NN:
    def __init__(self, iS=(20, 20, 11), lr=0.001):
        self.learningRate = lr
        self.inputShape = iS

        input_global = keras.Input(shape=(39, 39, 11), name='Game_state_global')
        input_local = keras.Input(shape=(3, 3, 11), name='Game_state_local')

        layer_local_0 = layers.Conv2D(16, 2, strides=1, activation="relu")(input_local)

        layer_global_0 = layers.Conv2D(32, 5, strides=2, activation="relu")(input_global)
        layer_global_1 = layers.Conv2D(32, 5, strides=2, activation="relu")(layer_global_0)

        layer_global_flat = layers.Flatten()(layer_global_1)
        layer_local_flat = layers.Flatten()(layer_local_0)

        f = layers.concatenate([layer_local_flat, layer_global_flat])

        f = layers.Dense(512, activation="relu")(f)

        output = layers.Dense(11, activation="softmax", name="Units_actions")(f)

        self.model = keras.Model(inputs=[input_global, input_local], outputs=output)

        self.model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=self.learningRate))
        print(self.model.summary())

    def load_model(self, filepath):
        self.model = load_model(filepath)
        return self.model
