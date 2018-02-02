import pandas as pd
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from results.viz import ImageGridCallback,BatchGridCallback
from keras.utils import multi_gpu_model
import time
from keras.utils.generic_utils import Progbar
from collections import defaultdict
import numpy as np
import os



class universal_train(object):
    def __init__(self,model,epochs,batch_size,ngpu,exp,MIMO=True):
        self.model=model
        self.MIMO=MIMO
        self.epochs=epochs
        self.batchsize=batch_size
        self.ngpu=ngpu
        self.tracking_names(exp)

    def tracking_names(self,exp):
        defaultDir='results/'
        def make_new(name):
            status='exist'
            if not os.path.exists(name):
                os.makedirs(name)
                status = 'new'
            return status
        dirname=os.path.join(os.getcwd(), defaultDir+exp.folderName+'/'+exp.vs)
        self.tmp_depth=dirname+"/prediction/depth_map"
        self.tmp_can = dirname + "/prediction/can_map"
        self.tmp_weights=dirname+"/trainWeights/"
        status=[make_new(i)for i in [dirname,self.tmp_depth,self.tmp_can,self.tmp_weights]]
        print('dir status:',status)



        self.train_hist_dir=dirname+'/'+exp.vs+"_train_history.csv"
        self.model_name = dirname + '/' + exp.vs+"_" + exp.model+"_.h5"
        self.weight_name= dirname + '/' + exp.vs+"_" + exp.model+"_weight_.hdf5"

    def train(self,x_train,y_train,x_test,y_test):
        print('Training model','.'*10)
        WeightCallbacks = ModelCheckpoint(filepath=self.tmp_weights+'{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss',
                            verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=5)

        ImageCallback = ImageGridCallback(xtest=x_test, ytest=y_test, model=self.model, can_folder=self.tmp_can,
                                     depth_folder=self.tmp_depth, VIO=self.MIMO)

        # self.model.load_weights('results/CNNGpu/0.01/0.01_SingleInputOutput_weight_.hdf5')
        # self.model = multi_gpu_model(self.model, self.ngpu) if self.ngpu > 1 else self.model
        self.model.compile(optimizer='adam', loss='mse')
        history = self.model.fit(x_train, y_train,
                                  epochs=self.epochs,
                                  batch_size=self.ngpu * self.batchsize,
                                  verbose=2,

                                  shuffle=True,
                                  validation_data=(x_test, y_test),
                                  callbacks=[ImageCallback, WeightCallbacks]
                                  )
        self.book_keeping(self.model,history.history)
    def batch_train(self,bxs,exp=None):
        print('Batch Training Started', '.' * 10)
        if(exp):
            self.tracking_names(exp)
        avg_train_loss=3
        avg_test_loss=3
        beta=0.7
        pbar=Progbar(bxs.batch_iteration)
        history=defaultdict(list)
        BatchImageCallback = BatchGridCallback(model=self.model, can_folder=self.tmp_can,
                                          depth_folder=self.tmp_depth, VIO=self.MIMO, every=5)

        self.model = multi_gpu_model(self.model, self.ngpu) if self.ngpu > 1 else self.model
        self.model.compile(optimizer='adam', loss='mse')


        def get_data(data):
            if self.MIMO:
                x_train = [data['depth_x_train'], data['can_x_train']]
                x_test = [data['depth_x_test'], data['can_x_test']]
                return (x_train,x_train),(x_test,x_test)
            else:
                return (data['depth_x_train'], data['can_x_train']),(data['depth_x_test'], data['can_x_test'])


        for e in range(self.epochs):
            st=time.time()

            for b in range(bxs.batch_iteration):
                (x_train, y_train), (x_test, y_test)=get_data(next(bxs))
                train_loss=self.model.train_on_batch(x_train,y_train)
                test_loss=self.model.test_on_batch(x_test,y_test)

                train_loss=sum(train_loss) if self.MIMO else train_loss
                test_loss=sum(test_loss) if self.MIMO else test_loss
                avg_train_loss=beta*avg_train_loss+(1-beta)*train_loss
                avg_test_loss = beta * avg_test_loss + (1 - beta) * test_loss
                pbar.update(b)

                print('\t train_loss {:.3f} test_loss {:.3f}'.format( avg_train_loss,
                                                                                    avg_test_loss))

            et=time.time()
            print('\n','*'*60)

            print('[{:.3f}] epoch {} train_loss {:.3f} test_loss {:.3f}'.format(et-st, e,avg_train_loss,avg_test_loss) )
            history['train_loss'].append(avg_train_loss)
            history['test_loss'].append(avg_test_loss)


            BatchImageCallback(e,x_test)

        self.book_keeping(self.model, history)

    def seq_train(self,bxs,batch_size,exp=None):
        print('Sequance Training Started', '.' * 10)
        if (exp):
            self.tracking_names(exp)
        avg_train_loss = 3
        avg_test_loss = 3
        beta = 0.7
        pbar = Progbar(bxs.batch_iteration//batch_size)
        history = defaultdict(list)

        self.model = multi_gpu_model(self.model, self.ngpu) if self.ngpu > 1 else self.model
        self.model.compile(optimizer='adam', loss='mse')




        def get_data(data):
            if self.MIMO:
                x_train = [data['depth_x_train'], data['can_x_train']]
                x_test = [data['depth_x_test'], data['can_x_test']]
                return (x_train, x_train), (x_test, x_test)
            else:
                return (data['depth_x_train'], data['can_x_train']), (data['depth_x_test'], data['can_x_test'])

        for e in range(self.epochs):
            st = time.time()

            for b in range(bxs.batch_iteration//batch_size):



                x_train, y_train= [], []
                x_test, y_test = [], []
                for i in range(batch_size):
                    (xx_train, yy_train), (xx_test, yy_test) = get_data(next(bxs))
                    x_train.append(xx_train)
                    y_train.append(yy_train)
                    x_test.append(xx_test)
                    y_test.append(yy_test)

                try:
                    train_loss = self.model.train_on_batch(np.array(x_train), np.array(y_train))
                    test_loss = self.model.test_on_batch(np.array(x_test), np.array(y_test))

                    train_loss = sum(train_loss) if self.MIMO else train_loss
                    test_loss = sum(test_loss) if self.MIMO else test_loss
                    avg_train_loss = beta * avg_train_loss + (1 - beta) * train_loss
                    avg_test_loss = beta * avg_test_loss + (1 - beta) * test_loss
                    pbar.update(b)

                    print('\t train_loss {:.3f} test_loss {:.3f}'.format(avg_train_loss,
                                                                         avg_test_loss))
                except:
                    print("error in epoch{} at batch{} ".format(e,b))







            et = time.time()
            print('\n', '*' * 60)

            print('[{:.3f}] epoch {} train_loss {:.3f} test_loss {:.3f}'.format(et - st, e, avg_train_loss,
                                                                                avg_test_loss))
            history['train_loss'].append(avg_train_loss)
            history['test_loss'].append(avg_test_loss)



        self.book_keeping(self.model, history)




    def book_keeping(self,model,log):


        # save complete model and weight
        # model.save(self.model_name)
        print("\n \n saving weights",'.'*60)
        model.save_weights(self.weight_name)
        print("\n \n  weights saved", '.' * 60)
        dtf = pd.DataFrame(log)
        dtf.to_csv(self.train_hist_dir)

