# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 22:15:19 2017

@author: HimaSkiran
"""

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.layers import Dense,Input,Conv2D,MaxPooling2D,UpSampling2D
from PIL import Image
import numpy as np
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt




def Preprocess_Data(InputData):
    InputData=InputData.reshape((len(InputData),28,28,1))
    InputData=InputData.astype('float32')/255.0   #Normalization of Input data
    return InputData

def Build_Encoder():
    input_image=Input(shape=(28,28,1))
    Conv1=Conv2D(32,(3,3),activation='relu',padding='same')(input_image)
    MaxP1=MaxPooling2D((2,2),padding='same')(Conv1)
    Conv2=Conv2D(16,(3,3),activation='relu',padding='same')(MaxP1)
    MaxP2=MaxPooling2D((2,2),padding='same')(Conv2)
    Conv3=Conv2D(16,(3,3),activation='relu',padding='same')(MaxP2)
    
   
    encoded=MaxPooling2D((2,2),padding='same')(Conv3)
    
    return input_image,encoded

def Build_Decoder(EncodedData):
    

    ConvT1=Conv2D(16,(3,3),activation='relu',padding='same')(EncodedData)
    
    
   
    US1=UpSampling2D((2,2))(ConvT1)
    ConvT2=Conv2D(16,(3,3),activation='relu',padding='same')(US1)
    
    US2=UpSampling2D((2,2))(ConvT2)
    ConvT3=Conv2D(32,(3,3),activation='relu',padding='valid')(US2) #Valid to match output with Input
    US3=UpSampling2D((2,2))(ConvT3)
    decoded=Conv2D(1,(3,3),activation='sigmoid',padding='same')(US3)
    return decoded

def Add_Noise(dataset,Noise):
    dataset=dataset+(Noise*np.random.normal(0,1,dataset.shape))  # Noise to data set
    return dataset


    
    

def main():
    (trainx,trainy),(testx,testy)=keras.datasets.mnist.load_data()
    print(trainy.shape)
    trainx=Preprocess_Data(trainx)
    testx=Preprocess_Data(testx)
    trainx_Noise=Add_Noise(trainx,0.3)
    testx_Noise=Add_Noise(testx,0.3)   
     
    Input_layer,Encoder=Build_Encoder()
    Decoder=Build_Decoder(Encoder)
    Autoenc=Model(Input_layer,Decoder)
    Autoenc.compile(loss='binary_crossentropy',optimizer='adadelta')
    Autoenc.summary()
    Autoenc.fit(trainx_Noise,trainx,epochs=40,batch_size=200,
                    shuffle=True)
    # %%
    res=Autoenc.predict(testx_Noise[100].reshape((1,28,28,1)))
    t=testx_Noise[100].reshape((28,28))
    plt.imshow(t,cmap='gray')
    plt.show()
    res=res.reshape((28,28))
    plt.imshow(res,cmap='gray')
    plt.show()

if __name__=='__main__':
    main()