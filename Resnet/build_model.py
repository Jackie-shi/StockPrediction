import Resnet_lstm as rs
from tensorflow.keras import layers, regularizers
from tensorflow import keras
from keras import backend as K

def new_sigmoid(x):
    return 1.7159 * K.tanh((2/3) * x)

def get_model():
    # inputs = keras.Input(shape=(2, 5, 37))
    # x = rs.ResNet([2, 2])
    # x = layers.LeakyReLU(alpha=0.7)(x)
    # x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l1(0.1))(x)
    # x = layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l1(0.1))(x)
    # x = layers.Flatten()(x)
    # x = layers.Dense(1)(x)
    # outputs = new_sigmoid(x)
    # model = keras.Model(output=outputs)
    # return model
    model = keras.Sequential(
        [
            rs.ResNet([2, 2]),
            layers.LeakyReLU(alpha=0.7),
            layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l1(0.1)),
            layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l1(0.1)),
            layers.Flatten(),
            layers.Dense(1, activation='sigmoid')
        ]
    )
    return model
