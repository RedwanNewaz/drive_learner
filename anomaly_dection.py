from keras.models import model_from_json
from keras.utils import multi_gpu_model
import numpy as np
from collections import defaultdict
from dataset.VideoLoader import ActivitesDict
import pandas as pd
import matplotlib.pyplot as plt
import json
import time

'''
#TEST:Anomaly Detection
1) detection time 0.0058 sec
2) AutoEncoder outperform CNN
'''


class model_testing(object):
    def __init__(self,model_name,weight_name,verbose=True,MIMO=True):
        self.model_name=model_name
        self.weight_name=weight_name
        self.verbose=verbose
        self.datasetSize=6624
        self.threshold=0.13
        self.MIMO=MIMO

    # 0.00551058
    def loadModel(self):
        with open(self.model_name, 'r') as f:
            json_string = json.load(f)
        model = model_from_json(json_string)
        model = multi_gpu_model(model, 2)
        model.load_weights(self.weight_name)
        print('weight loaded ', '.' * 60)
        model.compile(optimizer='adam', loss='mse')
        self.model=model

    def loadDataset(self,act1,act2):
        act=ActivitesDict()
        x,_=act[act1]
        _,y=act[act2]
        self.x_index=act(act1)
        self.y_index=act(act2)
        return x,y

    def contextual_anomalies(self,act1,act2,xtype='order',ytype='order',test_num=200):
        self.loadModel()
        x,y=self.loadDataset(act1,act2)
        rs = lambda x: np.reshape(x, [1] + list(x.shape))


        inx=np.random.randint(0,len(x),size=test_num) if xtype is 'rand' else np.arange(test_num)
        iny=np.random.randint(0,len(y),size=test_num) if ytype is 'rand' else np.arange(test_num)
        x=x[inx]
        y=y[iny]
        self.x_index=self.x_index[inx]
        self.y_index = self.y_index[inx]

        test_result = []
        for i, (_x, _y) in enumerate(zip(x,y)):
            if not self.MIMO:
                result = self.model.test_on_batch(rs(_x), rs(_y))
            else:
                xy=[rs(_x), rs(_y)]
                res=self.model.evaluate(xy,xy,verbose=0)
                result=sum(res)
            test_result.append(result)

        self.result_dict = {'frame':self.x_index.tolist(),'prediction': test_result}
        print("\n \n")
        print('*' * 10, 'Test Done', '*' * 10)
        print('max score {:.4f}'.format(max(test_result)))
        print('avg score {:.4f}'.format(np.mean(np.array(test_result))))




    def save(self,name):
        df = pd.DataFrame.from_dict(self.result_dict)
        df.to_csv(name)
        print('*' * 10, 'Log Saved', '*' * 10)

    def plot(self,name):

        xx,yy=self.result_dict['frame'],self.result_dict['prediction']
        plt.plot(xx,yy)



        # plt.hist(np.array(df))
        for i,(x,y) in enumerate(zip(xx,yy)):
            if y>self.threshold:
                y=np.squeeze(y)
                plt.scatter(x,y,c='r')
                anx=int(np.squeeze(self.y_index[i]))
                # print(anx)
                # plt.annotate(anx,(x,y))
                # print(x,y)

        plt.title("Anomaly Detection")
        plt.legend(['Prediction Loss','Anomaly'])
        # plt.axis([0,100,0,0.1])




        plt.xlabel("Index")
        plt.ylabel("Loss")
        ns=name.split('.c')
        plt.savefig(ns[0]+'.png')
        print('*' * 10, 'Plot Saved', '*' * 10)



# /home/guest/deepLearning/depth_can/results/AELargeImg/0.02


if __name__ == '__main__':
    cnn  = './summary/CNNLargeImg.json'
    weight_cnn = './results/CNNLargeImg/0.02/0.02_SISO_weight_.hdf5'
    autoencoder= './summary/AELargeImg.json'
    weight_autoencoder='./results/AELargeImg/0.02/0.02_MIMO_weight_.hdf5'
    expname='left_left_order'

    test=model_testing(model_name=autoencoder,weight_name=weight_autoencoder,verbose=True,MIMO=True)
    test.contextual_anomalies( 'Left','Left', 'order','order',test_num=100)


    # test.save('./results/AELargeImg/0.02/anomalies/%s.csv'%expname)

    test.plot('./results/AELargeImg/0.02/anomalies/%s.csv'%expname)






