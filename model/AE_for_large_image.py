from keras.layers import Input, Dense, Conv2D, MaxPooling2D,GlobalMaxPooling2D, UpSampling2D, BatchNormalization
from keras.layers import LeakyReLU,BatchNormalization,Reshape,Flatten,concatenate,Add,Dropout,Activation
from keras.models import Model


def encoder_model(x,power):

    def encode_unit(x1,h_dim):
        x1 = Conv2D(h_dim, (4, 4), activation='relu', padding='same')(x1)
        x1 = MaxPooling2D((2, 2), padding='same')(x1)
        x1 = Dropout(0.5)(x1)
        return x1

    for p in power:
        x = encode_unit(x, 2 ** p)
    else:
        x = Conv2D(16, (4, 4), activation='relu', padding='same')(x)
        x = GlobalMaxPooling2D()(x)
        enc = Dense(16, activation='softmax')(x)
    return enc

def shared_layers(encoder):
    x = Dense(5 * 129 * 16, activation='relu')(encoder)
    x = Dropout(0.5)(x)
    x = Reshape((5, 129, 16))(x)

    x = Conv2D(16, (4, 4), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Dropout(0.5)(x)

    x = Conv2D(32, (4, 4), activation='relu', padding='same')(x)
    x = GlobalMaxPooling2D()(x)
    return x

def depth_decoder(xx):
    xx = Dense(20 * 515, activation='relu')(xx)
    xx = Reshape((20, 515, 1))(xx)
    xx = Conv2D(128, (1, 1), activation='relu')(xx)
    xx = UpSampling2D((2, 2))(xx)
    xx = Dropout(0.5)(xx)
    decoder = Conv2D(1, (4, 4), activation='sigmoid', padding='same')(xx)
    return decoder

def can_decoder(xx):
    xx = Dense(8 * 50, activation='relu')(xx)
    xx = Reshape((8, 50, 1))(xx)
    xx = Conv2D(32, (1, 1), activation='relu')(xx)
    xx = UpSampling2D((2, 2))(xx)
    xx = Dropout(0.5)(xx)
    decoder = Conv2D(1, (4, 4), activation='sigmoid', padding='same')(xx)
    return decoder



if __name__ == '__main__':
    depth_img = Input(shape=(40, 1030, 1))
    can_img = Input(shape=(16, 100, 1))

    depth_encoder = encoder_model(x=depth_img, power=[7, 6, 5])
    can_encoder = encoder_model(x=can_img, power=[5])
    encoder = Add()([depth_encoder, can_encoder])

    xDepth, xCan = shared_layers(encoder), shared_layers(encoder)
    decoder = [depth_decoder(xDepth), can_decoder(xCan)]

    autoencoder = Model([depth_img, can_img], decoder)
    autoencoder.compile(loss='mse', optimizer='adam')


    autoencoder.summary()
