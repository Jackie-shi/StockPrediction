import tensorflow as tf
from matplotlib import pyplot as plt
import build_model
tf.compat.v1.disable_eager_execution()
from pandas import DataFrame
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense, \
    GlobalAveragePooling2D, TimeDistributed, ConvLSTM2D, Conv1D
import numpy as np
from tensorflow.keras import Model

train = np.load('/Users/jackieshi/PycharmProjects/pythonProject4/V5/V5_data/V5_data_3d.npy', allow_pickle=True)
# train_1 = np.load('predict_dataV5_1.npy', allow_pickle=True)


x_train = train[0][0]
y_train = train[0][1]
# x_train=np.concatenate((train[0][0],train_1[0][4][5800:6000,:,:,:]),axis=0)
# y_train=np.concatenate((train[0][1],train_1[0][5][5800:6000,:]),axis=0)
x_test = train[0][2]
y_test = train[0][3]
# x_eval = train_1[0][0]
# y_eval = train_1[0][5]
print(x_train.shape)
print(y_test.shape)
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
#
# gpus = tf.config.experimental.list_physical_devices('GPU')
# # 对需要进行限制的GPU进行设置
# tf.config.experimental.set_virtual_device_configuration(gpus[0],
#                                                         [tf.config.experimental.VirtualDeviceConfiguration(
#                                                             memory_limit=2048)])
'''from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=0)
X_resampled, y_resampled = ros.fit_sample(X, y)
sorted(Counter(y_resampled).items())'''

# import sys
# sys.exit()

class AI(Model):
    def __init__(self, strides=1):
        super(AI, self).__init__()

        self.strides = strides
        self.c1 = Conv1D(2, 3, input_shape=(1088, 5, 37), data_format='channels_first', strides=strides, padding='same',
                         use_bias=False, )
        self.c2 = Conv1D(2, 2, input_shape=(1088, 5, 37), data_format='channels_first', strides=strides, padding='same',
                         use_bias=False)
        self.b1 = BatchNormalization()
        self.a1 = tf.keras.layers.LeakyReLU(alpha=0.8)
        self.lstm = tf.keras.layers.LSTM(64, return_sequences=True)
        self.f1 = tf.keras.layers.Dense(10, kernel_regularizer=tf.keras.regularizers.l2())

    def call(self, inputs):
        print(inputs.shape)
        x1, x2 = tf.unstack(inputs, axis=1)
        print(x1.shape)
        x1 = self.c1(x1)
        x2 = self.c1(x2)
        x1 = self.c2(x1)
        x2 = self.c2(x2)

        # print(x.shape)
        # x = self.b1(x)
        # print(x.shape)
        # x = self.c2(x)
        # x1,x2= tf.unstack(x, axis=1)

        x1 = self.lstm(x1)
        x2 = self.lstm(x2)
        # x = self.p1(x)
        # print(x.shape)
        x = tf.stack([x1, x2], axis=1)
        y = self.f1(x)
        return y


model = build_model.get_model()

'''    
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dropout(0.25))

,model.add(tf.keras.layers.Dense(64, activation='relu'))
'''
# model.add(tf.keras.layers.LeakyReLU(alpha=0.7))
# model.add(tf.keras.layers.Dense(64, activation='relu', kernel_regularizer='l1'))
# model.add(tf.keras.layers.Dense(32, activation='relu'))
# model.add(tf.keras.layers.Flatten())
# model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.SGD(momentum=0.000001, learning_rate=0.008, decay=0.001),
              metrics=['accuracy']
              )

history = model.fit(x_train, y_train, batch_size=272, epochs=34, validation_data=(x_test, y_test), validation_freq=1,
                    steps_per_epoch=96, validation_steps=1)  # ,callbacks=[TensorBoard(log_dir='./tmp/log')])

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

# model.save('模型')
