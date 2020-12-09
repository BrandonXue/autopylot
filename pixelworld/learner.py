# Non-local modules
from tensorflow import keras
from tensorflow.keras import layers, models

class DeepQLearner:
    def __init__(self, env, input_shape):
        self.env = env
        self.input_shape = input_shape

    def __create_optimizer(self):
        # https://ruder.io/optimizing-gradient-descent/index.html#adam
        # Nesterov-accelerated Adaptive Moment Estimation
        optimizer = keras.optimizers.Nadam(
            learning_rate=0.001
        )
        return optimizer
    

    def __create_model(self):
        model = keras.Sequential([
            layers.Conv2D(
                8, 5, strides=(1, 1), activation='relu', 
                input_shape=self.input_shape,               # specified in constructor
                data_format='channels_last'                 # (w, h, channels)
            ),
            layers.Conv2D(
                16, 3, strides=(1, 1), activation='relu',
                data_format='channels_last'
            ),
            layers.Flatten(data_format='channel_last'),
            layers.Dense(units=256, activation='relu'),
            layers.Dense(units=4, activation='tanh')
        ])
        return model
