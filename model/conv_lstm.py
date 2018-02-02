from keras.models import Sequential
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Dropout,GlobalAveragePooling3D,LSTM,Dense,Reshape

# vs 0.01 we don't include convlstm2d filter 32<<16 16
# result :epoch 9 train_loss 0.066 test_loss 0.041

# vs 0.02  convlstm2d filter 32 is divided into 16 16
def get_model(seq_len=40):

    seq = Sequential()
    seq.add(ConvLSTM2D(filters=16, kernel_size=(4, 4),
                       input_shape=(seq_len, 40, 1030, 1),
                       padding='same', return_sequences=True,activation='relu'))
    seq.add(BatchNormalization())

    seq.add(ConvLSTM2D(filters=16, kernel_size=(4, 4),
                       padding='same', return_sequences=True,activation='relu'))
    seq.add(BatchNormalization())

    seq.add(Dropout(0.5))

    seq.add(ConvLSTM2D(filters=16, kernel_size=(4, 4),
                       padding='same', return_sequences=True,activation='relu'))
    seq.add(BatchNormalization())
    seq.add(Dropout(0.5))

    # seq.add(ConvLSTM2D(filters=128, kernel_size=(1, 1),
    #                    padding='same', return_sequences=True,activation='relu'))
    # seq.add(BatchNormalization())

    seq.add(GlobalAveragePooling3D())

    seq.add(Dense(1600, activation="relu"))

    seq.add(Reshape((1,16, 100, 1)))

    seq.add(Conv3D(filters=seq_len, kernel_size=(4, 4, 4),
                   activation='sigmoid',
                   padding='same', data_format='channels_last'))
    seq.add(Reshape((seq_len,16, 100, 1)))

    return seq
