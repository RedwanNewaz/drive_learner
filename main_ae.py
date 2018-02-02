import matplotlib
matplotlib.use('Agg')
import argparse
from model.train_any_model import universal_train
from dataset.VideoLoader import VideoLoader,VideoBatchLoader
from model.AE_for_large_image import *
import json



if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Experiment Details for DepthMap & CanMap Processing')

    args.add_argument('--folderName',type=str, default='AELargeImg')
    args.add_argument('--vs', type=str, default='0.02')
    args.add_argument('--model', type=str, default='MIMO')




# TODO: Build Autoencoder model
    depth_img = Input(shape=(40, 1030, 1))
    can_img = Input(shape=(16, 100, 1))

    depth_encoder = encoder_model(x=depth_img, power=[7, 6, 5])
    can_encoder = encoder_model(x=can_img, power=[5])
    encoder = Add()([depth_encoder, can_encoder])

    xDepth, xCan = shared_layers(encoder), shared_layers(encoder)
    decoder = [depth_decoder(xDepth), can_decoder(xCan)]

    autoencoder = Model([depth_img, can_img], decoder)



# TODO: save model as JSON format
    print('saving model as Json Format ','.'*10)
    json_string =autoencoder.to_json()
    with open('summary/AELargeImg.json','w+') as outfile:
        json.dump(json_string,outfile)


# TODO: Train model

    # vs=VideoLoader()
    # data=vs.get_data()
    # Xtrain = [data['depth_x_train'], data['can_x_train']]
    # Xtest = [data['depth_x_test'], data['can_x_test']]

    vbs=VideoBatchLoader()
    print(vbs)

    param = args.parse_args()
    mimo = param.model is "MIMO"

    train = universal_train(model=autoencoder, epochs=100, batch_size=32, ngpu=2, exp=param, MIMO=mimo)

    # train.train(x_train=Xtrain,
    #             y_train=Xtrain,
    #             x_test=Xtest,
    #             y_test=Xtest
    #             )
    train.batch_train(vbs)





