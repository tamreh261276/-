import os
import librosa   #for audio processing
import librosa.display
import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile #for audio processing
import warnings
warnings.filterwarnings("ignore")

from keras.models import load_model
model=load_model('best_model.hdf5')


def predict(audio):
    prob=model.predict(audio.reshape(1,8000,1))
    index=np.argmax(prob[0])
    return classes[index]
    
import random
index=random.randint(0,len(x_val)-1)
samples=x_val[index].ravel()
print("Audio:",classes[np.argmax(y_val[index])])
ipd.Audio(samples, rate=8000)
print("Text:",predict(samples))
#we will try the prediction over Positive sample that system should mark as true 
#note some sample will amrk as True as True Positive 
#and other will amrk as False as False Positive , samples that system should mark it as true but for some learning staff 
#it couldn't recoginze this so we can enhance learning later 
positive_audio_path = 'C:/Users/Ahmad/Desktop/QuranAI/Code/Predict/Positive/'
negative_audio_path = 'C:/Users/Ahmad/Desktop/QuranAI/Code/Predict/Negative/'

#Predicting the correct voices
 waves = [f for f in os.listdir(positive_audio_path + '/'+ label) if f.endswith('.wav')]
    for wav in waves:
        samples, sample_rate = librosa.load(filepath + '/' + 'stop.wav', sr = 16000)
        samples = librosa.resample(samples, sample_rate, 8000)
        ipd.Audio(samples,rate=8000)  
        predict(samples)

#Predicting the incorrect voices
 waves = [f for f in os.listdir(negative_audio_path + '/'+ label) if f.endswith('.wav')]
    for wav in waves:
        samples, sample_rate = librosa.load(filepath + '/' + 'stop.wav', sr = 16000)
        samples = librosa.resample(samples, sample_rate, 8000)
        ipd.Audio(samples,rate=8000)  
        predict(samples)
        
 #we use a binary classification for each Quran Aya if user speaking it correctly or not 