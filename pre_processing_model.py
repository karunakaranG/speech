import os
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from python_speech_features import mfcc
import numpy
"""
def mfcc(signal,samplerate=16000,winlen=0.025,winstep=0.01,numcep=13,
                 nfilt=26,nfft=512,lowfreq=0,highfreq=None,preemph=0.97,
     ceplifter=22,appendEnergy=True)

https://github.com/jameslyons/python_speech_features
"""
def normalize(data):
    data1=numpy.zeros(data.shape,dtype=float)
    maximum=numpy.max(abs(data))
    data1=data/maximum
    return data1


def process_y_label(label):
        final=np.zeros(10,dtype=float)
        la=eval(label)#convert string to int
        final[la]=1.0
        return final

    

DIR='C:/Users/KARUNAKARAN G/Desktop/signal_speech/speech/recordings/'

def preprocess():
    data_x_length=[]
    data=[]
    final_y_data=[]
    for img in os.listdir(DIR):
        
        label=img.split('.')[-1]
        if label=='wav':
            path=os.path.join(DIR,img)
            sample_rate,signal=wav.read(path)
            cep=mfcc(signal,sample_rate,winlen=0.030,winstep=0.020,numcep=13,nfilt=26,nfft=512,lowfreq=0,highfreq=None,preemph=0.97,ceplifter=22,appendEnergy=True)
            #plt.plot(cep.T)
            #plt.show()
            cep=cep.T#(13,.)
            data.append(cep)
            value=cep.shape[1]
            y_length=cep.shape[0]
            data_x_length.append(value)
            label=img.split('_')[-3]
            y_label=process_y_label(label)
            final_y_data.append(y_label)
                    
    x_length=np.max(np.array(data_x_length))
    final_x_data=[]
    final_data=[]
    print(y_length,x_length)
    for i in data:
        data_pad=np.zeros((y_length,x_length))
        value=np.array(i)
        x_axis=value.shape[0]
        y_axis=value.shape[1]
        data_pad[:x_axis,:y_axis] = value
        length=y_length*x_length
        data_pad=np.reshape(data_pad,(length))
        final_x_data.append(data_pad)

    return np.array(final_x_data),np.array(final_y_data)
    print(np.array(final_x_data).shape)
    print(np.array(final_y_data).shape)
    print(np.array(final_data).shape)
    
#print(np.array(data).shape)


