import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os,cv2

#matplotlib을 통해 보여줄 그래프의 크기를 직접 지정해 준것.
from pylab import rcParams
rcParams['figure.figsize'] = 20, 10

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import tensorflow as tf

from keras.utils import np_utils
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from keras.models import Sequential
from keras.layers import *
from keras.metrics import categorical_accuracy
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint
from keras.optimizers import *
from keras.layers import BatchNormalization

data_path = './Input/CK+48'
file_list = os.listdir(data_path)
file_list.remove('.DS_Store')
print("file list:",file_list)

# Any results you write to the current directory are saved as output

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.datasets import make_classification
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

import itertools

#디렉토리에서 이미지를 추출하기

# data_dir_list = os.listdir(data_path)

num_epoch=10

img_data_list=[]

# labels = np.ones((num_of_samples,),dtype='int64')
labels = []
labeled = 0
img_count = 0
#우리가 파일을 만든 순서가 아닌 컴퓨터가 지가 읽는 순서대로 파일을 불러옴.
for dataset in file_list:
    # if dataset in '.DS_Store':
    #         continue
    img_one_cat_list = []
    img_list=os.listdir(data_path+'/'+ dataset)
    print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
    for img in img_list:
        #이미지를 해당 경로에서 하나씩 읽어와서 (48,48)사이즈로 읽어와 img_data_list라는 array에 저장.        
        input_img=cv2.imread(data_path + '/'+ dataset + '/'+ img )
        input_img_resize=cv2.resize(input_img,(48,48),interpolation=cv2.INTER_AREA)
        img_data_list.append(input_img_resize)
        img_one_cat_list.append(input_img_resize)
    for label in range(len(img_one_cat_list)):
        labels.append(labeled)

    labeled += 1

img_data = np.array(img_data_list)
labels = np.array(labels)
print(labels.shape)
print(img_data.shape)
# print(img_data_list)
# img_data = np.array(img_data_list)
# img_data = img_data.astype('float32')
# img_data = img_data/255 #normalization


# #데이터 라벨링

# num_classes = 7

# #전체 이미지의 개수를 받아옴(981개)
# num_of_samples = img_data.shape[0]

# #샘플 이미지의 개수만큼 데이터가 1인 numpy array를 생성. shape : (981,)
# labels = np.ones((num_of_samples,),dtype='int64')

# # print(labels.shape)
# labels[0:134]=0 #135 anger:0 으로 라벨링
# labels[135:188]=1 #54 contempt:1 으로 라벨링
# labels[189:365]=2 #177 disgust:2 으로 라벨링
# labels[366:440]=3 #75 fear:3 으로 라벨링
# labels[441:647]=4 #207 happy:4 으로 라벨링
# labels[648:731]=5 #84 sadness:5 으로 라벨링
# labels[732:980]=6 #249 surprise:6 으로 라벨링
