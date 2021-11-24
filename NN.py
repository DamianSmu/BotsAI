from tensorflow import keras
from keras import layers
from keras.optimizer_v2.adam import Adam

import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
import keras.optimizers


class NN:
    def __init__(self, iS=(20, 20, 11), lr=0.0005):
        self.learningRate = lr
        self.inputShape = iS
        self.outputShape = 4

        inputs = keras.Input(shape=iS, name='Game state')
        f = layers.Flatten()(inputs)
        f = layers.Dense(iS[0] * iS[1], activation="sigmoid")(f)
        f = layers.Reshape((iS[0], iS[1], -1))(f)
        units = layers.Dense(12, activation="softmax", name="Units_actions")(f)
        # cities = layers.Dense(2, activation="sigmoid", name="Cities_actions")(f)
        # output = layers.Concatenate()([units, cities])
        output = units
        self.model = keras.Model(inputs=inputs, outputs=output)

        # Kompilowanie modelu
        self.model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=self.learningRate))

    # def __init__(self, iS=(100, 100, 3), lr=0.0005):
    #     self.learningRate = lr
    #     self.inputShape = iS
    #     self.numOutputs = 4
    #     self.model = Sequential()
    #
    #     # Dodawanie warstw do modelu
    #     self.model.add(Conv2D(32, (3, 3), activation='relu', input_shape=self.inputShape))
    #
    #     self.model.add(MaxPooling2D((2, 2)))
    #
    #     self.model.add(Conv2D(64, (2, 2), activation='relu'))
    #
    #     self.model.add(Flatten())
    #
    #     self.model.add(Dense(units=256, activation='relu'))
    #
    #     self.model.add(Dense(units=self.numOutputs))
    #
    #     # Kompilowanie modelu
    #     self.model.compile(loss='mean_squared_error', optimizer=Adam(lr=self.learningRate))

    # Utworzenie funkcji, która załaduje model z pliku
    def loadModel(self, filepath):
        self.model = load_model(filepath)
        return self.model
