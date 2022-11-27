import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os,cv2
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
from keras.callbacks import ModelCheckpoint
from keras.optimizers import *
import os
data_path = './Input/CK+48'
file_list = os.listdir(data_path)
file_list.remove('.DS_Store')
print("file list:",file_list)

# Any results you write to the current directory are saved as output

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.datasets import make_classification
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

#디렉토리에서 이미지를 추출하기

num_epoch=10

img_data_list=[]

labels = []
labeled = 0
img_count = 0
#우리가 파일을 만든 순서가 아닌 컴퓨터가 지가 읽는 순서대로 파일을 불러옴.
for dataset in file_list:
    img_one_cat_list = []
    img_list=os.listdir(data_path+'/'+ dataset)
    print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
    for img in img_list:
        #이미지를 해당 경로에서 하나씩 읽어와서 (48,48)사이즈로 읽어와 img_data_list라는 array에 저장.        
        input_img=cv2.imread(data_path + '/'+ dataset + '/'+ img )
        input_img_resize=cv2.resize(input_img,(48,48),interpolation=cv2.INTER_AREA)
        img_data_list.append(input_img_resize)
        img_one_cat_list.append(input_img_resize)
    #라벨링 넘파이 배열 생성 
    for label in range(len(img_one_cat_list)):
        labels.append(labeled)

    labeled += 1

img_data = np.array(img_data_list)
labels = np.array(labels)

#이미지 데이터 정규화

img_data = img_data.astype('float32')
img_data = img_data/255 #normalization
print(img_data.shape)

num_classes = len(file_list)

# 카테고리 리스트로 선언
names = file_list

#라벨 이름 리턴(인코딩)
def getLabel_Incoding(labels):
    labels = labels.tolist()
    return names[labels.index(1)]
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
# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

#모델 생성 및 요약
import creat_model as cm

model_custom = cm.create_model()

model_custom.summary()

#Conduct k-Fold Cross-validation

from sklearn.model_selection import KFold

# kfold 교차 검증
# 데이터가 적은 데이터 셋에 대하여 정확도를 향상 시키기 위함.
# 이는 기존에 Training / Validation / Test 세개의 집단으로 분류하는것 보다 Training / Test로만 분류할때 학습 데이터 셋이 더 많기 때문.

# 5개의 폴드 세트로 분리하는 K Fold 객체와 폴드 세트별 정확도를 담을 리스트 객체 생성.
kf = KFold(n_splits=5, shuffle=False)

#Data Augmentation 기능 / 데이터 증강 기법
#모델의 성능을 높이면서 오버피팅을 극복하기 위해 학습 데이터의 다양성을 늘리기 위함.
#하나의 원본 이미지를 다양항 버전으로 만들어 학습시키는 것.
#대표적인 데이터 증강 기법으로는 원본 이미지를 수평 또는 수직 반전 시키는 방법, 회전시키는 방법, 일부를 자르는 방법,RGB채널 순서를 바꾸거나 
#픽셀값을 변경하여 원본 이미지의 색조를 변경시키는 것들이 있음.
#이러한 이미지 증강 기법을 tf.keras의 ImageDataGenerator 가 대신해줌.
#이 툴 이외에 Data Albumenation이나, ImgAug, Tensorflow Image Library 등 다양한 데이터 증강 기법 툴들이 있음.

from keras.preprocessing.image import ImageDataGenerator

# 이미지 증강법 1단계.ImageDataGenerator 객체(aug)를 생성하면서 증강을 수행할 유형들을 지정.
aug = ImageDataGenerator(
    rotation_range=25, width_shift_range=0.1,
    height_shift_range=0.1, shear_range=0.2, 
    zoom_range=0.2,horizontal_flip=True, 
    fill_mode="nearest")


#Training Model 
BS = 8 # batch size
EPOCHS = 200 # epochs

result = []
scores_loss = []
scores_acc = []
k_no = 0

#Kfold 객체의 split()을 호출하면 폴드별 학습용, 검증용 테스트의 로우 인덱스를 array로 반환
#split()이 알아서 학습용 검증용 데이터를 나누어줌. 위에서 k=5라고 지정한 경우 for문이 5번 돌아 가면서 학습용, 검증용 데이터를 나누어 줌.
for train_index, test_index in kf.split(x):
    #Kfold 객채의 split()으로 반환된 인덱스를 이용해 학습용, 검증용 테스트 데이터를 추출함.
    X_Train,X_Test = x[train_index], x[test_index]
    Y_Train,Y_Test = y[train_index], y[test_index]

    # 최적의 모델을 저장할 경로
    file_path = "./weights_best_emotion_detect.hdf5"
    # 콜백 함수. (체크포인트, 조기 종료 선언)
    checkpoint = ModelCheckpoint(file_path, monitor='loss', save_best_only=True, mode='min')
    early = EarlyStopping(monitor="loss", mode="min", patience=8)

    callbacks_list = [checkpoint, early]

    #이미지 증강법 2단계 aug.flow(X,y,batch_size=,shuffle=) : X(훈련 이미지),y(라벨), 이미지를 한번에 몇개 업로드 시킬지에 대한 배치 사이즈,데이터 셔플 유무
    #flow 메소드로 생성한 객체는 Numpy Array Iterator 객체로 우리가 흔히 아는 Python Iterator처럼 loop문이나 next()함수를 사용해 Iterator 안에 데이터를 하나씩 호출 가능.

    hist = model_custom.fit_generator(aug.flow(X_Train, Y_Train), epochs=EPOCHS,validation_data=(X_Test, Y_Test), callbacks=callbacks_list)
    # model.fit(X_Train, Y_Train, batch_size=batch_size, epochs=epochs, validation_data=(X_Test, Y_Test), verbose=1)
    
    # model_custom.load_weights(file_path)
    result.append(model_custom.predict(X_Test))
    score = model_custom.evaluate(X_Test,Y_Test)
    scores_loss.append(score[0])
    scores_acc.append(score[1])
    k_no+=1

print(scores_acc,scores_loss)    