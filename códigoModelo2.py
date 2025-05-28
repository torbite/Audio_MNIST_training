import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split


data_dir = 'MNIST_DATA'


sample_rate = 22050  
n_mfcc = 13  
max_pad_len = 128


def add_noise(data, noise_factor=0.005):
    noise = np.random.randn(len(data))
    return data + noise_factor * noise


def audio_to_mfcc(audio, sr=sample_rate, max_pad_len=max_pad_len):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    if mfcc.shape[1] < max_pad_len:
        pad_width = max_pad_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_pad_len]
    return mfcc


X, y = [], []
for speaker_dir in os.listdir(data_dir):
    speaker_path = os.path.join(data_dir, speaker_dir)
    if os.path.isdir(speaker_path):
        for file in os.listdir(speaker_path):
            if file.endswith('.wav'):
                label = int(file.split('_')[0])  
                file_path = os.path.join(speaker_path, file)

            
                audio, sr = librosa.load(file_path, sr=sample_rate)  
                mfcc = audio_to_mfcc(audio)  
                X.append(mfcc)
                y.append(label)

                
                noisy_audio1 = add_noise(audio)
                noisy_audio2 = add_noise(audio)
                noisy_audio3 = add_noise(audio)
                noisy_audio4 = add_noise(audio)


                
                X.append(audio_to_mfcc(noisy_audio1, sr))
                X.append(audio_to_mfcc(noisy_audio2, sr))
                X.append(audio_to_mfcc(noisy_audio3, sr))
                X.append(audio_to_mfcc(noisy_audio4, sr))
                y.extend([label, label, label, label])  

X = np.array(X)  
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = X_train[..., np.newaxis] 
X_test = X_test[..., np.newaxis] 

input_shape = (n_mfcc, max_pad_len, 1)  

print(input_shape)

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax') 
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


history = model.fit(X_train, y_train, epochs=18, batch_size=32, validation_data=(X_test, y_test))


test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f'Test accuracy: {test_acc:.2f}')

model.save('audio_mnist_model3.h5')  
