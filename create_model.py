from keras.models import Sequential
from keras.layers import *

#모델 만들기
def create_model():
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