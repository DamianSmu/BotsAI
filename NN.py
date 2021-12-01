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

        inputs = keras.Input(shape=iS, name='Game state')
        layer0 = layers.Conv2D(32, 4, strides=2, activation="relu")(inputs)
        layer1 = layers.Conv2D(64, 4, strides=2, activation="relu")(layer0)

        f = layers.Flatten()(layer1)
        f = layers.Dense(iS[0] * iS[1], activation="relu")(f)
        f = layers.Reshape((iS[0], iS[1], -1))(f)

        output = layers.Dense(12, activation="softmax", name="Units_actions")(f)

        self.model = keras.Model(inputs=inputs, outputs=output)

        self.model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=self.learningRate))
        print(self.model.summary())

    def load_model(self, filepath):
        self.model = load_model(filepath)
        return self.model
