from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D,GlobalMaxPooling2D
from keras.layers import LeakyReLU,BatchNormalization,Reshape,Flatten,concatenate,Add,Dropout,Activation
from keras.models import Model
from keras import backend as K
K.set_image_data_format('channels_last')

def largeImgCNN(input_layer):
    x = Conv2D(16, (4, 4), activation='relu')(input_layer)  # single stride 4x4 filter for 16 maps
    x = Conv2D(32, (4, 4), activation='relu')(x)  # single stride 4x4 filter for 32 maps
    x = Dropout(0.5)(x)
    x = Conv2D(64, (4, 4), activation='relu')(x)  # single stride 4x4 filter for 64 maps
    x = Dropout(0.5)(x)
    x = Conv2D(128, (1, 1))(x)  # finally 128 maps for global average-pool
    x = GlobalMaxPooling2D()(x)  # pseudo-dense 128 layer
    x = Dense(1600, activation="sigmoid")(x)  # softmax output
    output_layer = Reshape((16, 100,1))(x)
    return output_layer

