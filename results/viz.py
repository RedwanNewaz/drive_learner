import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import Callback
from keras.models import  load_model



def image_grid(imgfiles,filename):
    img_shape=np.shape(imgfiles)
    n = img_shape[0]
    def random_shuffule(files):
        # print('random shuffling display')
        indices=np.random.randint(10*10,size=100)
        f, axarr = plt.subplots(10, 10)
        indx=0
        for i in range(10):
            for j in range(10):
                rand_indx=indices[indx]
                img=np.reshape(files[rand_indx],(img_shape[1],img_shape[2]))
                plt.gray()

                axarr[i, j].imshow(img)
                axarr[i, j].axis('on')
                axarr[i, j].get_xaxis().set_visible(False)
                axarr[i, j].get_yaxis().set_visible(False)
                indx+=1

        plt.savefig('{}.png'.format(filename))
        plt.close('all')

    def ordered_img(files):
        # print('ordered display')
        n = img_shape[0]
        indices = np.arange(n)
        n1=int(np.sqrt(n))

        f, axarr = plt.subplots(n1, n1,figsize=(15,15))
        indx = 0
        for i in range(n1):
            for j in range(n1):
                rand_indx = indices[indx]
                img = np.reshape(files[rand_indx], (img_shape[1], img_shape[2]))
                plt.gray()

                axarr[i, j].imshow(img)
                axarr[i, j].axis('on')
                axarr[i, j].get_xaxis().set_visible(False)
                axarr[i, j].get_yaxis().set_visible(False)
                indx += 1

        plt.savefig('{}.png'.format(filename))
        plt.close('all')
    return random_shuffule(imgfiles) if n>100 else ordered_img(imgfiles)

class ImageGridCallback(Callback):
    num_epoch = 0
    num_disp = 3*3
    def __init__(self,xtest,ytest,model,can_folder,depth_folder,every=5,VIO=True):
        self.xsamples=xtest

        self.model=model
        self.every=every
        self.can_folder=can_folder+"/epoch-{}"
        self.depth_folder=depth_folder+"/epoch-{}"
        self.VIO=VIO #variable input and output ?

    def on_epoch_end(self, epoch, logs={}):
        self.num_epoch += 1
        if self.num_epoch%self.every==0:
            return self.vioPredict(epoch) if self.VIO else self.SIOpredict(epoch)

    def SIOpredict(self,epoch):
        depthmap  = self.xsamples
        randomIndex = np.random.randint(len(depthmap) - 1, size=self.num_disp)
        xRandSample = depthmap[randomIndex]
        can_img = self.model.predict(xRandSample)
        print("epoch {}  canShape {}".format(epoch, np.shape(can_img)))
        image_grid(can_img, self.can_folder.format(epoch))

    def vioPredict(self,epoch):
        depthmap, canmap = self.xsamples
        randomIndex = np.random.randint(len(depthmap) - 1, size=self.num_disp)
        xRandSample = [depthmap[randomIndex], canmap[randomIndex]]
        depth_img, can_img = self.model.predict(xRandSample)
        print("epoch {} depthShape {} canShape {}".format(epoch, np.shape(depth_img), np.shape(can_img)))
        can_image_path = self.can_folder.format(epoch)
        depth_image_path = self.depth_folder.format(epoch)
        # print("depth filename ",depth_image_path)
        image_grid(depth_img, depth_image_path)
        image_grid(can_img, can_image_path)



class BatchGridCallback(ImageGridCallback):

    def __init__(self,model,can_folder,depth_folder,every=5,VIO=True):
        self.model=model
        self.can_folder = can_folder + "/epoch-{}"
        self.depth_folder = depth_folder + "/epoch-{}"
        self.VIO = VIO  # variable input and output ?
        self.every = every
    def __call__(self,epoch,samples):
        self.xsamples = samples
        self.on_epoch_end(epoch+1)


if __name__ == '__main__':
    #load model
    model =load_model('./complete_model/cnn_model.h5')
    model.summary()
    #load model weight
    model.load_weights('./complete_model/cnn_weights.hdf5')
    #model predict
    xRandSample=np.random.uniform(0,1,(2,128,128,1))
    can_img = model.predict(xRandSample)
    print("predicted shape {}".format(np.shape(can_img)))






