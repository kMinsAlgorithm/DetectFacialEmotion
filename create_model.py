from keras.models import Sequential
from keras.layers import *

#모델 만들기(128*128)
def create_model_128():
    '''
    이는 공식 문서에서 128*128크기의 이미지를 학습하기 적합한 cnn 모델의 구조
    '''
    input_shape=(128,128,3)
 
    model = Sequential()
    model.add(Conv2D(16, (3, 3), input_shape=input_shape, padding='same', activation = 'relu'))
    model.add(Conv2D(16, (3, 3), input_shape=input_shape, padding='same', activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3), input_shape=input_shape, padding='same', activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), input_shape=input_shape, padding='same', activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3),padding='same', activation = 'relu'))
    model.add(Conv2D(64, (3, 3),padding='same', activation = 'relu'))
    model.add(Conv2D(64, (3, 3),padding='same', activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation = 'relu'))
    model.add(Dense(4, activation = 'softmax'))

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'],optimizer='adam')
    
    return model

#모델 만들기(224*224)
def create_model_224():
    '''
    이는 공식 문서에서 224*224크기의 이미지를 학습하기 적합한 cnn 모델의 구조
    '''
    input_shape=(224,224,3)
 
    model = Sequential()
    model.add(Conv2D(64, (5, 5), input_shape=input_shape, padding='same', activation = 'relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(96, (5, 5), input_shape=input_shape, padding='same', activation = 'relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3),padding='same', activation = 'relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3),padding='same', activation = 'relu'))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(4096, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2048, activation = 'relu'))
    model.add(Dense(5, activation = 'softmax'))

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'],optimizer='adam')
    
    return model