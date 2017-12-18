import matplotlib.pyplot as plt
from scipy import signal
import math, numpy
import scipy.io.wavfile as wav
import sounddevice as sd

fs,audio=wav.read("file.wav")

samp_rate = fs
sim_time = 60
nsamps = samp_rate*sim_time
cuttoff_freq = 0.1

#xfreq = numpy.fft.fft(audio)
#fft_freqs = numpy.fft.fftfreq(nsamps, d=1./samp_rate)

norm_pass = cuttoff_freq/(samp_rate/2)
norm_stop = 1.5*norm_pass
(N, Wn) = signal.buttord(wp=norm_pass, ws=norm_stop, gpass=2, gstop=30, analog=0)
(b, a) = signal.butter(N, Wn, btype='high', analog=0, output='ba')

#(w, h) = signal.freqz(b, a)

y = signal.lfilter(b, a, audio)
sd.play(audio,fs)
plt.plot(y)
plt.show()


































##import pyaudio
##import scipy.io.wavfile as wav
##import wave
##import sounddevice as sd
##import matplotlib.pyplot as plt
##import numpy as np
##import scipy.signal as signal
##
##
##
##
##
##value=signal.savgol_filter(audio,7,7)
##plt.plot(value)
##plt.show()
##value=np.fft.fft(audio)
##print(value.shape)
##mea=-1.1*(np.mean(value,axis=0))
##print(mea[1],mea[0])
##sha=audio.shape
##print(sha[0])
##for i in range(0,sha[0]):
##    for j in range(0,1):
##        if (audio[i,j]<=mea[1] and audio[i,j]>=mea[0]):
##            audio[i,j]=0
##            
##        
###sd.play(audio)
##plt.plot(audio)
##plt.show()
