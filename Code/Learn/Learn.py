import os
import librosa   #for audio processing
import librosa.display
import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile #for audio processing
import warnings
warnings.filterwarnings("ignore")

#Data Exploration and Visualization


train_audio_path = 'C:/Users/Ahmad/Desktop/QuranAI/Code/Learn/DataSet/'
fullpath = train_audio_path+'1/learnsample.wav'
samples, sample_rate = librosa.load(fullpath)
fig = plt.figure(figsize=(14, 8))
ax1 = fig.add_subplot(211)
ax1.set_title('Raw wave of ' + fullpath)
ax1.set_xlabel('time')
ax1.set_ylabel('Amplitude')

# Generates the graphic
librosa.display.waveplot(samples, sr=sample_rate, ax=ax1)
# Prints plots
plt.show()

print('Sampling rate: ' + str(sample_rate))
print('Sample number: ' + str(len(samples)))
#ax1.plot(np.linspace(0, sample_rate/len(samples), sample_rate), samples)


ipd.Audio(samples, rate=sample_rate)

samples = librosa.resample(samples, sample_rate, 8000)
ipd.Audio(samples, rate=8000)

labels=os.listdir(train_audio_path)

#find count of each label and plot bar graph
no_of_recordings=[]
for label in labels:
    print(label)
    waves = [f for f in os.listdir(train_audio_path + '/'+ label) if f.endswith('.wav')]
    no_of_recordings.append(len(waves))
    
#plot
plt.figure(figsize=(1,5))
index = np.arange(len(labels))
plt.bar(index, no_of_recordings)
plt.xlabel('Quranic verse', fontsize=12)
plt.ylabel('No of recordings', fontsize=12)
plt.xticks(index, labels, fontsize=15, rotation=60)
plt.title('No. of recordings for each verse')
plt.show()

#we give each Quran Aya and Id as Labeling , for sample model we provide only 6 Aya till we have a big data set contain all Quran Ayat
labels=["1", "2", "3", "4","5","6"]

#we load all avalible Sample for each Aya and we save evey aya samples in a different folder 
duration_of_recordings=[]
for label in labels:
    waves = [f for f in os.listdir(train_audio_path + '/'+ label) if f.endswith('.wav')]
    for wav in waves:
        sample_rate, samples = wavfile.read(train_audio_path + '/' + label + '/' + wav)
        duration_of_recordings.append(float(len(samples)/sample_rate))
    
plt.hist(np.array(duration_of_recordings))





all_wave = []
all_label = []
for label in labels:
    print(label)
    waves = [f for f in os.listdir(train_audio_path + '/'+ label) if f.endswith('.wav')]
    for wav in waves:
        samples, sample_rate = librosa.load(train_audio_path + '/' + label + '/' + wav, sr = 16000)
        samples = librosa.resample(samples, sample_rate, 8000)
        if(len(samples)== 8000) : 
            all_wave.append(samples)
            all_label.append(label)

#we use some preprocessing funtionality to minmize noise , adjust samplerate and other staff
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y=le.fit_transform(all_label)
classes= list(le.classes_)

from keras.utils import np_utils
y=np_utils.to_categorical(y, num_classes=len(labels))

#for evaluation of our model we try to dicive our learning samples into two section
#fisrt section provide to learn Quran Reading Rules Auto correction
#Secon section provide to Evaluate Quran Reading Rules Auto correction while learning and check Learning result metric to guide how model quality reach
all_wave = np.array(all_wave).reshape(-1,8000,1)
from sklearn.model_selection import train_test_split
x_tr, x_val, y_tr, y_val = train_test_split(np.array(all_wave),np.array(y),stratify=y,test_size = 0.2,random_state=777,shuffle=True)

#Model building and build our neural network layers as each layer will has it's own filteration and weights
from keras.layers import Dense, Dropout, Flatten, Conv1D, Input, MaxPooling1D
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
K.clear_session()

inputs = Input(shape=(8000,1))

#First Conv1D layer
conv = Conv1D(8,13, padding='valid', activation='relu', strides=1)(inputs)
conv = MaxPooling1D(3)(conv)
conv = Dropout(0.3)(conv)

#Second Conv1D layer
conv = Conv1D(16, 11, padding='valid', activation='relu', strides=1)(conv)
conv = MaxPooling1D(3)(conv)
conv = Dropout(0.3)(conv)

#Third Conv1D layer
conv = Conv1D(32, 9, padding='valid', activation='relu', strides=1)(conv)
conv = MaxPooling1D(3)(conv)
conv = Dropout(0.3)(conv)

#Fourth Conv1D layer
conv = Conv1D(64, 7, padding='valid', activation='relu', strides=1)(conv)
conv = MaxPooling1D(3)(conv)
conv = Dropout(0.3)(conv)

#Flatten layer
conv = Flatten()(conv)

#Dense Layer 1
conv = Dense(256, activation='relu')(conv)
conv = Dropout(0.3)(conv)

#Dense Layer 2
conv = Dense(128, activation='relu')(conv)
conv = Dropout(0.3)(conv)

outputs = Dense(len(labels), activation='softmax')(conv)

model = Model(inputs, outputs)
model.summary()

#we have more than optimizer but for now we select adam as it give more accurate over orcoseeing time 
#some paramter need a well training and this showen by the time and some test experience 
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

#EarlyStopping call back use to stop learning if the model reach the quality needed in early time no need to waiting it to be finished 
#since we reach our target early 
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, min_delta=0.0001) 
#we save only the best check point for now since we use a small model

mc = ModelCheckpoint('best_model.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')

history=model.fit(x_tr, y_tr ,epochs=100, callbacks=[es,mc], batch_size=32, validation_data=(x_val,y_val))

#show some model statics and losses to guide the model quality 
from matplotlib import pyplot 
pyplot.plot(history.history['loss'], label='train') 
pyplot.plot(history.history['val_loss'], label='test') 
pyplot.legend() pyplot.show()
