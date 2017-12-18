from pre_processing_model import preprocess
import numpy as np
import tflearn
import sounddevice as sd
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from python_speech_features import mfcc

x_data,y_data=preprocess()
print(x_data.shape)
print(y_data.shape)

input_size=x_data.shape[1]
classes=y_data.shape[1]
length_data=x_data.shape[0]

def shuffle_content(num,data,labels):
    idx=np.arange(0,len(data))
    np.random.shuffle(idx)
    idx=idx[:num]
    data_shuffle=[data[i] for i in idx]
    labels_shuffle=[labels[i] for i in idx]
    return np.asarray(data_shuffle),np.asarray(labels_shuffle)

X_data,Y_data=shuffle_content(length_data,x_data,y_data)

train_x_data=X_data[:-300]
train_y_data=Y_data[:-300]
test_x_data=X_data[-300:]
test_y_data=Y_data[-300:]

# Building deep neural network
input_layer = tflearn.input_data(shape=[None,input_size])
dense1 = tflearn.fully_connected(input_layer, 500, activation='relu')
dropout1 = tflearn.dropout(dense1, 0.8)

dense2 = tflearn.fully_connected(dropout1, 500, activation='relu')
dropout2 = tflearn.dropout(dense2, 0.8)

#dense3 = tflearn.fully_connected(dense2, 100, activation='sigmoid',regularizer='L2', weight_decay=0.001)
#dropout3 = tflearn.dropout(dense3, 0.8)

softmax = tflearn.fully_connected(dropout2, 10, activation='softmax')

# Regression using SGD with learning rate decay and Top-3 accuracy
sgd = tflearn.SGD(learning_rate=0.001, lr_decay=0.96, decay_step=1000)
top_k = tflearn.metrics.Top_k(3)



net = tflearn.regression(softmax, optimizer=sgd, metric=top_k,
                         loss='categorical_crossentropy')

# Training
model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(train_x_data,train_y_data, n_epoch=30, validation_set=(test_x_data,test_y_data), snapshot_step=500, show_metric=True, run_id="dense model")
#model.save('speech_recognition_numbers.tflearn')
##print(val_val)
##va=np.argmax(model.predict(val))
##print(va)



path='C:/Users/KARUNAKARAN G/Desktop/signal_speech/speech/recordings/4_jackson_41.wav'
sample_rate,signal=wav.read(path)
sd.play(signal,sample_rate)
cep=mfcc(signal,sample_rate,winlen=0.030,winstep=0.020,numcep=13,nfilt=26,nfft=512,lowfreq=0,highfreq=None,preemph=0.97,ceplifter=22,appendEnergy=True)
cep=cep.T
y_length=13
x_length=57
data_pad=np.zeros((13,57))
x_axis=cep.shape[0]
y_axis=cep.shape[1]
data_pad[:x_axis,:y_axis] = cep
length=y_length*x_length
data_pad=np.reshape(data_pad,(length))
final_data=[]
final_data.append(data_pad)
input_size=data_pad.shape[0]
print(data_pad.shape)

print(np.argmax(model.predict(final_data)))
