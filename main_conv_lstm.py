import matplotlib
matplotlib.use('Agg')
import argparse
from model.train_any_model import universal_train
from dataset.VideoLoader import VideoLoader,VideoBatchLoader,VariableDataset
from model.conv_lstm import *
import json
import numpy as np
from keras.utils import multi_gpu_model




if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Experiment Details for DepthMap & CanMap Processing')

    args.add_argument('--folderName',type=str, default='RNN_CNN_Learning')
    args.add_argument('--vs', type=str, default='0.02')
    args.add_argument('--model', type=str, default='SISO')



    batch_size = 4
    seq_length = 20

    lstm_cnn_model=get_model(seq_length)



    # TODO: Train model in Batch mode
    vbs=VideoBatchLoader(batch_size=seq_length)

    print(vbs)

    param = args.parse_args()
    mimo = param.model is "MIMO"
    # TODO: save model as JSON format
    print('saving model as Json Format ', '.' * 10)
    json_string = lstm_cnn_model.to_json()
    with open('summary/%s.json'%param.folderName, 'w+') as outfile:
        json.dump(json_string, outfile)

    train = universal_train(model=lstm_cnn_model, epochs=10, batch_size=seq_length, ngpu=2, exp=param, MIMO=mimo)

    foldernames = [
        "ELECOM_20160319_123014_133014",
        "ELECOM_20160319_134804_144304",
        "ELECOM_20160319_163019_172519"
    ]

    versions = ['60.01', '60.02', '60.03']

    for f, v in zip(foldernames, versions):
        datasampler = VariableDataset(f)
        param.vs = v
        train.seq_train(vbs, batch_size,param)


