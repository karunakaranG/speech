from get_audio import get_audio_model
import numpy as np
import tflearn
import sounddevice as sd
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from python_speech_features import mfcc
from noise_remove import noise_remove_model
get_audio_model()
noise_remove_model()
path='C:/Users/KARUNAKARAN G/Desktop/projects/speech/filtered.wav'
fs,audio=wav.read(path)
##audio=audio[50000:]
##audio2=audio[:20000]
plt.plot(audio)
plt.show()

signal=audio
signal = signal[0:int(0.5 * fs)]
audio = np.append(signal[0], signal[1:] - 0.97 * signal[:-1])

plt.plot(audio)
plt.show()

audio1=[]
inde=0
for i in audio:
    if (abs(i)>10000):
        
        break;
    inde=inde+1
audio1=audio[(inde-10000):]
inde=0
audio4= audio1[::-1]
audio2=[]
for i in audio4:
    if (abs(i)>10000):
        break;
    inde=inde+1
audio2=audio4[(inde-10000):]
audio2= audio2[::-1]
plt.plot(audio2)
plt.show()
sd.play(audio2,fs)

signal=audio2
sample_rate=fs
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

#Building deep neural network
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
#model.fit(train_x_data,train_y_data, n_epoch=100, validation_set=(test_x_data,test_y_data), snapshot_step=500, show_metric=True, run_id="dense model")
#model.save('speech_recognition_numbers.tflearn')
#print(val_val)
model.load('speech_recognition_numbers.tflearn',weights_only=True)
va=np.argmax(model.predict(final_data))
print(va)
