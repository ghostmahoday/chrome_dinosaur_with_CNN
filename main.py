import pyautogui as pag
import matplotlib.pyplot as plt
import time
import cv2
import numpy as np
import keyboard
import os
import sys
from datetime import datetime
import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D, Dropout, LeakyReLU
import tensorflow as tf
from tensorflow import keras

print('starting')
time.sleep(5)
print('started')

for i in range(10000):
    time.sleep(0.01)
    region = (580,160,200,215)
    img= pag.screenshot(region=region)
    now = datetime.now()
    seconds = str(now.strftime("%d%m%Y%H%M%S"))
    name = '0'+ seconds + str(i)+'.jpg'
    
    if (keyboard.is_pressed('up')):
        name = '1'+ seconds + str(i)+'.jpg'
        
    if (keyboard.is_pressed('q')):
        sys.exit()
        
    img.save('./images/'+name)
   
files = os.listdir('./images')    
x=[]
y=[]

for i in files:
    img= cv2.imread('./images/'+i,0)
    img = cv2.resize(img, (64,64))
    x.append(img)
    y.append(int(i[0]))

data = pd.DataFrame({'image':x, 'label':y})
plt.imshow(data['image'][0])
data['image'][0].shape

data['label'].value_counts()

data_0 = data[data['label']==0]
data_1 = data[data['label']==1]

min_value = min(data_0.shape[0], data_1.shape[0])
data_0 = data_0.sample(min_value)

data = pd.concat([data_0, data_1])
data = data.sample(n=len(data), random_state=42)


data.columns
x = data['image']
y = data['label']


x = np.array(x)
y =  np.array(y)
x = np.array( [i.reshape(64,64,1) for i in x] )


model = Sequential()
model.add(Conv2D(32, (3,3), input_shape=(64,64,1), activation='relu'))
model.add(MaxPool2D((2,2)))
model.add(LeakyReLU(alpha=0.2))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPool2D((2,2)))
model.add(LeakyReLU(alpha=0.2))
model.add(Conv2D(128, (3,3), activation='relu'))
# model.add(MaxPool2D((2,2)))
model.add(LeakyReLU(alpha=0.2))
model.add(Conv2D(256, (3,3), activation='relu'))
# model.add(MaxPool2D((2,2)))
model.add(LeakyReLU(alpha=0.2))
model.add(Conv2D(512, (3,3), activation='relu'))
model.add(MaxPool2D((2,2)))
model.add(LeakyReLU(alpha=0.2))
# model.add(Conv2D(256, (3,3), activation='relu'))
# model.add(MaxPool2D((2,2)))
# model.add(LeakyReLU(alpha=0.2))
# model.add(Conv2D(32, (3,3), activation='relu'))
# model.add(MaxPool2D((2,2)))
model.add(Flatten())
model.add(Dense(2, activation='softmax'))

model.compile(optimixer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x,y, validation_split=0.2,epochs=40)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.show()

model.save('mushi_mushi.h5')

model = keras.models.load_model('mushi_mushi.h5')


time.sleep(5)

for i in range(10000):
    time.sleep(0.01)
    region = (580,160,730,215)
    img= pag.screenshot(region=region)
    cv2.imwrite('img.jpg',np.array(img))
    img = cv2.imread('./img.jpg', 0)
    img = cv2.resize(img,(64,64))
    img = np.array(img.reshape(1,64,64,1))
    img = tf.cast(img, dtype=tf.float16)
    ypred =model.predict(img)
    y_val = np.argmax(ypred)

    if y_val==1:
        print('up')
        pag.press('up')
    

time.sleep(5)
img= pag.screenshot(region=(580,160,200,215))
plt.imshow(np.array(img))    




