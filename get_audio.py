import pyaudio
import scipy.io.wavfile as wav
import wave
import sounddevice as sd
import matplotlib.pyplot as plt
def get_audio_model():
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    CHUNK = 1024
    RECORD_SECONDS = 5
    WAVE_OUTPUT_FILENAME = "file.wav"
 
    audio = pyaudio.PyAudio()
 
    # start Recording
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                rate=RATE, input=True,
                frames_per_buffer=CHUNK)
    print ("recording...")
    frames = []
 
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    print ("finished recording")
 
 
    # stop Recording
    stream.stop_stream()
    stream.close()
    audio.terminate()
 
    waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()
##
###get_audio()
##fs,audio=wav.read("output.wav")
##print(audio.shape)
##value=audio[52000:]
##value=value[:29000]
###value=audio[52000:,:]
###plt.plot(audio[:,0])
###plt.plot(audio[:,1])
###plt.show()
##
##sd.play(value,fs) #55000 to 78000
##plt.plot(value)
##plt.show()
####value1=value[:23000,:]
####sd.play(value1,fs)
###plt.plot(value1)
###plt.show()
