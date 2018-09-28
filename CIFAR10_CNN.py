# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 09:23:35 2017

@author: HimaSkiran
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv2D,MaxPooling2D,Flatten,Dropout,AveragePooling2D
import matplotlib.pyplot as plt
import numpy as np


with tf.device('/gpu:0'):   #/cpu:0


    def Build_Model(data_shape):
        Model=keras.Sequential()
        Model.add(Conv2D(64,kernel_size=(3,3),activation='relu',input_shape=data_shape))
        
        Model.add(Conv2D(64,kernel_size=(3,3),activation='relu',padding='same'))
      
        Model.add(AveragePooling2D(pool_size=(2,2),padding='same'))
        Model.add(Conv2D(128,kernel_size=(3,3),activation='relu',padding='same'))
        Model.add(Conv2D(128,kernel_size=(1,1),activation='relu',padding='same'))
        Model.add(AveragePooling2D(pool_size=(2,2),padding='same'))
        
        Model.add(Conv2D(128,kernel_size=(3,3),activation='relu',padding='same'))
        Model.add(Conv2D(128,kernel_size=(1,1),activation='relu',padding='same'))
        Model.add(AveragePooling2D(pool_size=(2,2),padding='same'))
        
        #Model.add(MaxPooling2D(pool_size=(2,2),padding='same'))
        
        Model.add(Flatten())
        Model.add(Dense(1000,activation='relu'))
        Model.add(Dropout(0.3))
        Model.add(Dense(512,activation='relu'))
        Model.add(Dropout(0.3))
        #Model.add(Dense(500,activation='relu'))
        Model.add(Dense(10,activation='softmax'))
        
        return Model
    
    def main():
        (TrainingX,TrainingY),(TestX,TestY)=keras.datasets.cifar10.load_data()
        classes=['Airplane','Automobile','Bird','Cat','Deer','Dog','Frog','Horse','Ship','Truck']
        TrainingY= keras.utils.to_categorical(TrainingY,num_classes=10)
        TestY=keras.utils.to_categorical(TestY,num_classes=10)
        TrainingX=TrainingX.astype('float32')/255.0
        TestX=TestX.astype('float32')/255.0
        Data_Aug=keras.preprocessing.image.ImageDataGenerator(rotation_range=20,height_shift_range=0.2,width_shift_range=0.2,horizontal_flip=True)
        Data_Aug.fit(TrainingX)
        
        
        
        Model=Build_Model((32,32,3))
        Model.summary()
        Model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
        Early_Stop=[keras.callbacks.EarlyStopping(monitor='val_loss',patience=10)]
        Model.fit(TrainingX,TrainingY,epochs=20,batch_size=100,validation_data=(TestX,TestY),callbacks=Early_Stop)
        #Model.fit_generator(Data_Aug.flow(TrainingX,TrainingY,batch_size=64),epochs=50,validation_data=(TestX,TestY),callbacks=Early_Stop)
        
        # %%
        res=Model.predict(TrainingX[0:2])
        plt.imshow(TrainingX[1],cmap='gray')
        plt.show()
        print(classes[ np.argmax(res[1])])
        print(classes[np.argmax(TrainingY[1])])
        
        
        Model.evaluate(TestX,TestY,verbose=1)
    
    if __name__=='__main__':
        main()