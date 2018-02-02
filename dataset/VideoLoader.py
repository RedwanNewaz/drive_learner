import skvideo.io
import numpy as np
import itertools
from tqdm import tqdm
import pandas as pd
from collections import defaultdict





def data_resize( frame):
    frame=np.array(frame)
    frame = frame.reshape((3, frame.shape[1],-1))
    frame = 0.299 * frame[0] + 0.587 * frame[1] + 0.114 * frame[2]
    frame =frame/255.
    return frame.T

class VideoLoader(object):
    def __init__(self,dataset_size=6643,batch_size=32,sub_folder=False):
        self.batch_size=batch_size
        self.dataset_size=dataset_size
        self.offset_index =(3*60+40)*10 #(3 min 40 sec)*10fps

        parent='./'if sub_folder is False else '../'

        self.depth_files=parent+'dataset/video/depth_video.avi'
        self.can_files=parent+'dataset/video/can_video.avi'
        self.resizing_data=lambda shape:tuple( list(shape)+[1])

    def read_video(self,name):
        videogen = skvideo.io.vread(name)
        return (data_resize(frame) for frame in tqdm(videogen[self.offset_index:]))


    def get_data(self):

        raw_depth_files = skvideo.io.vread(self.depth_files)
        raw_can_files =  skvideo.io.vread(self.can_files)
        depth_files=list(map(data_resize,raw_depth_files))
        can_files = list(map(data_resize, raw_can_files))

        print("get data depth{} can {}".format(np.shape(depth_files),np.shape(can_files)))


        dataset={}


        dataset['depthMap']= np.array(list(depth_files))
        dataset['canMap'] = np.array(list(can_files))
        #adding channel information to the datashape
        dataset['depthMapShape']=self.resizing_data(np.shape(dataset['depthMap']))
        dataset['canMapShape'] = self.resizing_data(np.shape(dataset['canMap']))
        # reshaping dataset
        dataset['depthMap']=np.reshape(dataset['depthMap'],dataset['depthMapShape'])
        dataset['canMap'] = np.reshape(dataset['canMap'], dataset['canMapShape'])


        train_percentage = 0.8  # How many % of data are for training?
        self.num_train=int(dataset['depthMapShape'][0] * train_percentage)
        (dataset['depth_x_train'], dataset['depth_x_test']) = np.split(dataset['depthMap'], [self.num_train])
        (dataset['can_x_train'], dataset['can_x_test']) = np.split(dataset['canMap'], [self.num_train])

        s=lambda x:np.shape(x)


        print('-'*61)
        ROW_FMT = '{0:<22s} | {}'
        print(ROW_FMT.format('Total depth Map',*s(dataset['depthMap'])))
        print(ROW_FMT.format('Total CAN Map', *s(dataset['canMap'])))

        print(ROW_FMT.format('depth_x_train', *s(dataset['depth_x_train'])))
        print(ROW_FMT.format('can_x_train', *s(dataset['can_x_train'])))

        print(ROW_FMT.format('depth_x_test', *s(dataset['depth_x_test'])))
        print(ROW_FMT.format('can_x_test', *s(dataset['can_x_test'])))
        print('-' * 61)
        self.dataset=dataset


        return self.dataset

class VideoBatchLoader(object):
    depth_files = 'dataset/video/depth_video.avi'
    can_files = 'dataset/video/can_video.avi'
    total_data = 9442


    count = 0
    dataset={}


    def __init__(self, subfolder=False,batch_size=32):
        parent = './' if subfolder is False else '../'
        self.batch_size = batch_size
        self.num_iter = self.total_data // batch_size + 1
        self.num_iter_train = int(self.num_iter * 0.8)
        self.num_iter_test = self.num_iter - self.num_iter_train
        self.batch_iteration = self.num_iter_train

        self._files=[parent+self.depth_files,parent+self.can_files]


        self.dataset=self.batch_processing()
    def __repr__(self):
        info='*'*60
        info+='\n'+'*'*20+'Video Batch Loader'+'*'*22
        info+="\n total data \t\t\t :{} \n batch size \t\t\t :{}\n num iteration \t\t\t :{}\n".format(self.total_data, self.batch_size,self.num_iter)
        info+=" num train iteration \t :{}\n num test iteration \t :{}\n".format(self.num_iter_train,self.num_iter_test)
        info += '*' * 60
        return info



    def get_batch(self,_files):

        # read video
        _raw_files = skvideo.io.vreader(_files)

        # split video
        gen_train_files = (itertools.islice(_raw_files, self.batch_size) for _ in range(self.num_iter_train))
        gen_test_files = (itertools.islice(_raw_files, self.batch_size) for _ in range(self.num_iter_test))


        return gen_train_files,gen_test_files

    def batch_processing(self):
        # print(self.num_iter_train,self.num_iter_test)
        dataset={}
        dataset['depth_x_train'], dataset['depth_x_test']=self.get_batch(self._files[0])
        dataset['can_x_train'], dataset['can_x_test']=self.get_batch(self._files[1])
        return dataset

    def update_dict(self,train_names,test_names):
        self.count += 1

        if self.count % self.num_iter_test == 0:
            temp_dataset = self.batch_processing()
            for key in test_names:
                self.dataset[key] = temp_dataset[key]

        if self.count % self.num_iter_train == 0:
            temp_dataset = self.batch_processing()
            for key in train_names:
                self.dataset[key] = temp_dataset[key]



    def __next__(self):

        get_value= lambda xx:[data_resize(bp)for bp  in xx]
        re_shape = lambda  xx: xx.reshape(tuple(list(xx.shape)+[1]))
        train_names=['depth_x_train', 'can_x_train']
        test_names = ['depth_x_test', 'can_x_test' ]
        names=train_names+test_names
        batch_items={}
        for key in names:

            _batch=next(self.dataset[key])
            batch_items[key]=get_value(_batch)
            batch_items[key] =re_shape(np.array(batch_items[key]))
        self.update_dict(train_names,test_names)


        return batch_items

class VariableDataset(VideoBatchLoader):
    total_data = 15510
    count = 0
    dataset={}
    def __init__(self,dirname,batch_size=32):
        self.depth_files = './dataset/ELECOM_HDD/%s/depth_video.avi'%dirname
        self.can_files = './dataset/ELECOM_HDD/%s/can_video.avi'%dirname
        self.batch_size = batch_size
        self.num_iter = self.total_data // batch_size + 1
        self.num_iter_train = int(self.num_iter * 0.8)
        self.num_iter_test = self.num_iter - self.num_iter_train
        self.batch_iteration = self.num_iter_train

        self._files=[self.depth_files,self.can_files]


        self.dataset=self.batch_processing()



class ActivitesDict(object):
    depth_files = './dataset/video/depth_video.avi'
    can_files = './dataset/video/can_video.avi'
    labels = './dataset/video/labeling.csv'

    def __init__(self):
        pass
    def group(self,filename):

        activities = pd.read_csv(self.labels, header=0)
        activities = activities.groupby('endFrame')
        video=skvideo.io.vreader(filename)

        video_frame=defaultdict(list)
        video_index=defaultdict(list)
        start=0
        for end, id in activities:
            key=np.array(id[['Activities']])
            key=str(np.squeeze(key))
            end=int(np.squeeze(np.array(end)))
            # frame=[start,end]
            frame=itertools.islice(video,start,end)
            video_frame[key].append(frame)
            video_index[key].append(np.arange(start,end))
            start=end+1

        self.video_index=video_index

        return video_frame

    def __call__(self, item):
        print('index ', item,)
        data=self.video_index[item]


        attribute=[]
        for d in data:
            attribute+=d.tolist()
        print(np.shape(attribute))
        return np.array(attribute)



    def __getitem__(self, item):
        self.depth_video = self.group(self.depth_files)
        self.can_video = self.group(self.can_files)

        depth_frames=[data_resize(a) for act in self.depth_video[item] for a in act ]
        can_frames = [data_resize(a) for act in self.can_video[item] for a in act]

        rs=lambda x:np.reshape(np.array(x),tuple(list(np.shape(x))+[1]))
        depth_frames,can_frames=rs(depth_frames),rs(can_frames)

        print(item," : ", np.shape(depth_frames), np.shape(can_frames))
        return depth_frames,can_frames


    def __repr__(self):

        keys="Straight \t Right \t Left \t stop \t SlowBrake \t Reverse"

        return keys



if __name__ == '__main__':
    ad=ActivitesDict()
    # getattr(ad)
    ad['Reverse']
    ad('Reverse')



