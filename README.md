# Ano_drive_detector in Keras
In my other repository, I have talked how to generate CAN image and Depth image.
The aim of this project is to learn driving behavior.
I have designed 3 neural networks -CNN, AUTOENCODER, LSTM-CNN to 
learn driving behavior from human drivers.

## CNN Architecture
<img src="https://github.com/RedwanNewaz/drive_learner/blob/master/model/covnet.png" alt="" data-canonical-src="https://gyazo.com/eb5c5741b6a9a16c692170a41a49c858.png" width="600" height="300" />

## AutoEncoder Architecture
<img src="https://github.com/RedwanNewaz/drive_learner/blob/master/model/autoencoder.png" alt="" data-canonical-src="https://gyazo.com/eb5c5741b6a9a16c692170a41a49c858.png" width="600" height="300" />

## LSTM-CNN Architecture
<img src="https://github.com/RedwanNewaz/drive_learner/blob/master/model/lstm_cnn.png" alt="" data-canonical-src="https://gyazo.com/eb5c5741b6a9a16c692170a41a49c858.png" width="600" height="300" />

The input(s) of this network is Depth image or both CAN and depth image. The output
depends on the type of deep learning algorithm we are using. 
For instance, the output of CNN is CAN image whereas input is depth image. 
Since the autoencoder takes multiple inputs -CAN image and depth image, 
yields multiple outputs as well. Finally, LSTM-CNN takes a squence of depth image and 
yileds a sequence of CAN image. For the convenient, we record the depth image and CAN image 
as a video file format using opencv library. 

## Dataset is empty!
Sorry floks! Driving dataset are not included in this repository.
However, if you have access to LiDAR sensor and CAN bus
of your autonomous car, we can easily create your own dataset using my rosbag decoder. 
Moreover, don't forget to record your driving data in rosbag file. Since we need to record multi-modality sensor outputs, 
ROSBAG is the most convenient format to deal with such type of data. 

## How to train?
Training of my models is very easy. There are three main files in the root direcory- main_cnn, main_ae, main_conv_lstm.
I have tried to reduce the parameters for training as low as possible. Let's take a look an example to train the CNN.


**Example: Training CNN**
```python
import argparse
from keras.models import model_from_json
from model.train_any_model import universal_train
from dataset.VideoLoader import VideoLoader,VideoBatchLoader,VariableDataset
from model.CNNLargeImage import *

args.add_argument('--folderName',type=str, default='cnn_result')
args.add_argument('--vs', type=str, default='25.01')
args.add_argument('--model', type=str, default='SISO')

param = args.parse_args()
mimo = param.model is "MIMO"

#Loading dataset
vs=VideoLoader()
data=vs.get_data()

# Loading existing model
model_name = './summary/CNNLargeImg.json'
with open(model_name, 'r') as f:
            json_string = json.load(f)
cnn_model = model_from_json(json_string)

#Training 
train = universal_train(model=cnn_model, epochs=10, batch_size=32, ngpu=2, exp=param, MIMO=mimo)
train.train(x_train=data['depth_x_train'],
                  y_train=data['can_x_train'],
                  x_test=data['depth_x_test'],
                  y_test=data['can_x_test']
                  )
```
There are 3 ways to load your training and testing dataset. If your dataset is small in size, you can use VideoLoader class to load it. For a large dataset, it is highly recommend to use VideoBatchLoader class to avoid resource exhausted error! Finally, the VariableDataset class can cope with the large video files located at different folders.
The model argument takes two type of input- either SISO (single input single output) or MIMO (multiple input multiple output). As I explained earlier, in case of autoencoder it would be MIMO whereas SISO for CNN/LSTM-CNN. Finally, the result of training saved inside the result direcory. I have assigned the options of choosing the folder name and version so that we can easily organize our results.
