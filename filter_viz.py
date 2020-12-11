
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from math import ceil, floor

class Visualizer:
    def __init__(self, model_name, target_model_name) -> None:
        self.loaded = True
        try:
            self.model = load_model(model_name)
        except:
            print('Model not loaded.')
            self.loaded = False
        else:
            print('Model loaded successfully.')

        try:
            self.model_target = load_model(target_model_name)
        except:
            print('Model not loaded.')
            self.loaded = False
        else:
            print('Target model loaded successfully.')


    def summarize(self):
        print('Model')
        print(self.model.summary())

        print('\n\nTarget Model')
        print(self.model_target.summary())


    def show_filters(self, layer_index):
        filters, biases = self.model.layers[layer_index].get_weights()

        min_val, max_val = filters.min(), filters.max()
        val_range = max_val - min_val

        normalized_filters = (filters - min_val) / val_range

        width, height, input_channels, feature_maps = normalized_filters.shape
        print(normalized_filters.shape)
        ncols = 8   # 8 columns
        nrows = ceil(feature_maps / 8) # As many rows as needed

        # Iterate over each filter and plot it
        for i in range(feature_maps):
            conv_filter = filters[:, :, :, i]
            ax = plt.subplot(nrows, ncols, i+1) # index must start at 1, not 0
            ax.set_xticks([])
            ax.set_yticks([])
            plt.imshow(conv_filter[:, :, 0], cmap='gray')
        plt.show()

            

