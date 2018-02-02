import matplotlib
matplotlib.use('Agg')
import argparse
from model.train_any_model import universal_train
from dataset.VideoLoader import VideoLoader,VideoBatchLoader,VariableDataset
from model.CNNLargeImage import *
import json
import numpy as np



if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Experiment Details for DepthMap & CanMap Processing')

    args.add_argument('--folderName',type=str, default='CNNLargeImg')
    args.add_argument('--vs', type=str, default='25.01')
    args.add_argument('--model', type=str, default='SISO')


    #TODO: Build model for large image
    depth_img = Input(shape=(None, None, 1))
    depth_encoder=largeImgCNN(depth_img)
    cnn_model = Model(inputs=[depth_img], outputs=[depth_encoder])


# TODO: save model as JSON format
    print('saving model as Json Format ','.'*10)
    json_string =cnn_model.to_json()
    with open('summary/CNNLargeImg.json','w+') as outfile:
        json.dump(json_string,outfile)






# # TODO: implement image grid visualization for tensorflowboard
#
#     vs=VideoLoader()
#     data=vs.get_data()
#
#     param = args.parse_args()
#     mimo = param.model is "MIMO"
#
#     train = universal_train(model=cnn_model, epochs=100, batch_size=32, ngpu=2, exp=param, MIMO=mimo)
#
#
#     train.train(x_train=data['depth_x_train'],
#                  y_train=data['can_x_train'],
#                  x_test=data['depth_x_test'],
#                  y_test=data['can_x_test']
#                  )
#
#




# TODO: Train model in Batch mode
#
#     vbs=VideoBatchLoader()
#     print(vbs)
#
    param = args.parse_args()
    mimo = param.model is "MIMO"

    train = universal_train(model=cnn_model, epochs=10, batch_size=32, ngpu=2, exp=param, MIMO=mimo)

#     train.batch_train(vbs)


    foldernames=[
        "ELECOM_20160319_123014_133014",
        "ELECOM_20160319_134804_144304",
        "ELECOM_20160319_163019_172519"
    ]


    versions=['60.01', '60.02','60.03']

    for f,v in zip(foldernames,versions):
        datasampler=VariableDataset(f)
        param.vs=v
        train.batch_train(datasampler,param)

