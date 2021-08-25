import tensorflow as tf
from matplotlib import pyplot as plt
from WindPy import w
import pandas as pd
tf.compat.v1.disable_eager_execution()
from pandas import DataFrame
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense,GlobalAveragePooling2D,TimeDistributed,ConvLSTM2D
from tensorflow.keras import Model
import numpy as np
from tensorflow.keras.callbacks import TensorBoard
import Resnet_lstm as rs

train=np.load('V5_data_3d.npy',allow_pickle=True)
train_1=np.load('pre_dataV4.npy',allow_pickle=True)
x_train=tf.convert_to_tensor(train[0][0])
y_train=tf.convert_to_tensor(train[0][1])
x_test=tf.convert_to_tensor(train[0][2])
y_test=tf.convert_to_tensor(train[0][3])
x_predict=tf.convert_to_tensor(train[0][4])
'''x_train=train[0][0]
y_train=train[0][1]
x_test=train[0][2]
y_test=train[0][3]
x_predict=train[0][4]'''
print(x_test.shape)

physical_devices = tf.config.experimental.list_physical_devices('GPU')

gpus = tf.config.experimental.list_physical_devices('GPU')
# 对需要进行限制的GPU进行设置
tf.config.experimental.set_virtual_device_configuration(gpus[0],
                                                        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])

'''from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=0)
X_resampled, y_resampled = ros.fit_sample(X, y)
sorted(Counter(y_resampled).items())'''
#print(y_train,y_eval,y_test)
#x_train = np.expand_dims(x_train, axis=3)
#x_test = np.expand_dims(x_test, axis=3)
#y_train = np.expand_dims(x_train, axis=3)
#y_test = np.expand_dims(x_test, axis=3)
model = tf.keras.models.Sequential()
#model.add(tf.keras.layers.ConvLSTM2D(64,kernel_size=2, return_sequences=True,activation='relu'))
#model.add(tf.keras.layers.Dropout(0.2))
model.add(rs.ResNet([2, 2, 2]))
#model.add(tf.keras.layers.Conv2D(128,data_format='channels_first',kernel_size=2,strides=1,activation='relu',input_shape=(2, 5, 37)))
#model.add(tf.keras.layers.MaxPool1D(pool_size=2,strides=1,padding='valid'))
#model.add(tf.keras.layers.Dropout(0.2))
#model.add(tf.keras.layers.Conv2D(32,data_format='channels_first',kernel_size=2,strides=1,activation='relu',input_shape=(2, 5, 37)))
#model.add(tf.keras.layers.Dropout(0.15))

#model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dropout(0.2))

#model.add(tf.keras.layers.LSTM(64, return_sequences=True, activation='relu'))
#model.add(tf.keras.layers.Dropout(0.2))
#model.add(tf.keras.layers.GRU(64, activation='relu'))
#model.add(tf.keras.layers.Dropout(0.25))

'''    
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dropout(0.25))

model.add(tf.keras.layers.Dense(64, activation='relu'))
'''
model.add(tf.keras.layers.Dense(64, activation='relu',kernel_regularizer='l1'))
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))


model.compile(loss='binary_crossentropy',
              optimizer='Adam',
              metrics=['accuracy']
              )

history = model.fit(x_train, y_train, batch_size=68, epochs=1000,validation_data=(x_test,y_test), validation_freq=1,steps_per_epoch=10,validation_steps=4)#,callbacks=[TensorBoard(log_dir='./tmp/log')])

'''y=model.predict(x_predict)
print('预测结果（y>1为风险警示）')
y=list(y)
#print(y)
file=open('his.txt','w')
his=[]
for i in y:
    i=list(i)
    #print(i)
    a=str(i[0])
    his.append(round(float(a)))
print(len(his))
writer=pd.ExcelWriter('history_compileV4.xlsx')
(pd.DataFrame(his)).to_excel(writer,sheet_name='result')
writer.save()
writer.close()
'''


model.summary()
print(model.summary())

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()