import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os,cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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
from keras.optimizers import *
from keras.layers import BatchNormalization
from keras.optimizers import Adam , RMSprop
from keras.callbacks import EarlyStopping ,ReduceLROnPlateau ,ModelCheckpoint

import os
data_path = 'Face&EmotionDetectUsingYolo/Input/CK+48'
file_list = os.listdir(data_path)
file_list.remove('.DS_Store')
print("file list:",file_list)

# Any results you write to the current directory are saved as output

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.datasets import make_classification
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


#디렉토리에서 이미지를 추출하기

data_dir_list = os.listdir(data_path)

num_epoch=10

img_data_list=[]


#우리가 파일을 만든 순서가 아닌 컴퓨터가 지가 읽는 순서대로 파일을 불러옴.
for dataset in data_dir_list:
    if dataset in '.DS_Store':
            continue
    img_list=os.listdir(data_path+'/'+ dataset)
    print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
    for img in img_list:
        #이미지를 해당 경로에서 하나씩 읽어와서 (48,48)사이즈로 읽어와 img_data_list라는 array에 저장.        
        input_img=cv2.imread(data_path + '/'+ dataset + '/'+ img )
        #input_img=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        input_img_resize=cv2.resize(input_img,(48,48))
        img_data_list.append(input_img_resize)
        
img_data = np.array(img_data_list)
img_data = img_data.astype('float32')
img_data = img_data/255 #normalization
print(img_data.shape)

#데이터 라벨링

num_classes = 7

#전체 이미지의 개수를 받아옴(981개)
num_of_samples = img_data.shape[0]

#샘플 이미지의 개수만큼 데이터가 1인 numpy array를 생성. shape : (981,)
labels = np.ones((num_of_samples,),dtype='int64')

# print(labels.shape)
labels[0:134]=0 #135 anger:0 으로 라벨링
labels[135:188]=1 #54 contempt:1 으로 라벨링
labels[189:365]=2 #177 disgust:2 으로 라벨링
labels[366:440]=3 #75 fear:3 으로 라벨링
labels[441:647]=4 #207 happy:4 으로 라벨링
labels[648:731]=5 #84 sadness:5 으로 라벨링
labels[732:980]=6 #249 surprise:6 으로 라벨링

# 카테고리 리스트로 선언
names = file_list

#argmax방식으로 리턴    
def getLabel_ByArgmax(preds):
    index = np.argmax(preds)
    return names[index]
    

# 훈련, 검증 데이터 셋 분리.

#원핫 인코딩 /  (파라미터 값, 0으로된 배열의 크기)
# 0: 1000000
# 1: 0100000
# 2: 0010000
#하나의 이미지에 대해 0과 1로 라벨링된(원핫인코딩된)배열을 리턴받음.
Y = np_utils.to_categorical(labels, num_classes)

#Shuffle the dataset
x,y = shuffle(img_data,Y, random_state=2)
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

#모델 생성 및 요약
import creat_model as cm

model_custom = cm.create_model()

model_custom.summary()

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rotation_range = 40,
    zoom_range = 0.2,
    horizontal_flip = True,
    fill_mode = 'nearest')

train_generator = train_datagen.flow(X_train,y_train,batch_size=8,shuffle=True)

model=Sequential([
                  Conv2D(64,3,activation='relu',kernel_initializer='he_normal',input_shape=(64,64,3)),
                  MaxPooling2D(3),
                  Conv2D(128,3,activation='relu',kernel_initializer='he_normal'),
                  Conv2D(256,3,activation='relu',kernel_initializer='he_normal'),
                  MaxPooling2D(3),
                  Flatten(),
                  Dense(256,activation='relu'),
                  Dense(7,activation='softmax',kernel_initializer='glorot_normal')
                  
])

es = EarlyStopping(
    monitor='val_accuracy', min_delta=0.0001, patience=10, verbose=2,
    mode='max', baseline=None, restore_best_weights=True
)
lr = ReduceLROnPlateau(
    monitor='val_accuracy', factor=0.1, patience=5, verbose=2,
    mode='max', min_delta=1e-5, cooldown=0, min_lr=0
)

callbacks = [es, lr]

model.summary()

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

history = model.fit(train_generator, batch_size=8 , epochs=20, validation_data = (X_train, y_train), callbacks = [callbacks],shuffle=True)