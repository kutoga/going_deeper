from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle

from keras.datasets import mnist
from keras.layers import Dropout, Convolution2D, MaxPool2D, Input, Activation, Flatten, Dense
import keras

import numpy as np

from ginfty import TDModel, GInftlyLayer, GammaRegularizedBatchNorm, c_l2

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape the data
num_classes = np.prod(np.unique(y_train).shape)
print("num_classes={}".format(num_classes))
data_shape = x_train[0].shape
if len(data_shape) < 3:
    data_shape += (1,)
x_train = x_train.reshape(x_train.shape[0], *data_shape)
x_test = x_test.reshape(x_test.shape[0], *data_shape)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
init_cnn_count = 30
assert init_cnn_count % data_shape[-1] == 0
init_cnn_repeat_factor = init_cnn_count // data_shape[-1]

# Build the model
fc_units = 256
w_step = 5
f_reg = 1e-8
w_reg = 5e-5
model = TDModel()
model += Input(data_shape)
model += Convolution2D(init_cnn_count, (3, 3), padding='same', trainable=False)
model += GInftlyLayer(
    'dcnn0',
    w_regularizer=(c_l2, w_reg),
    f_regularizer=(c_l2, f_reg),
    reweight_regularizer=False,
    f_layer=[
        lambda reg: Convolution2D(init_cnn_count, (3, 3), padding='same'),
        lambda reg: Dropout(0.25),
        lambda reg: GammaRegularizedBatchNorm(reg, max_free_gamma=0.),
    ], h_step=[
        lambda reg: Activation('relu'),
    ],
    w_step=w_step,
)
model += MaxPool2D()
model += Convolution2D(init_cnn_count * 2, (3, 3), trainable=False, padding='same')
model += GInftlyLayer(
    'dcnn1',
    w_regularizer=(c_l2, w_reg),
    f_regularizer=(c_l2, f_reg),
    reweight_regularizer=False,
    f_layer=[
        lambda reg: Convolution2D(init_cnn_count * 2, (3, 3), padding='same'),
        lambda reg: Dropout(0.25),
        lambda reg: GammaRegularizedBatchNorm(reg, max_free_gamma=0.),
    ], h_step=[
        lambda reg: Activation('relu'),
    ],
    w_step=w_step,
)
model += MaxPool2D()
model += Flatten()
model += Dense(fc_units, trainable=False)
model += GInftlyLayer(
    'dfc0',
    w_regularizer=(c_l2, 1e-3),
    f_regularizer=(c_l2, f_reg),
    reweight_regularizer=False,
    f_layer=[
        lambda reg: Dense(fc_units),
        lambda reg: Dropout(0.25),
        lambda reg: GammaRegularizedBatchNorm(reg, max_free_gamma=0.),
    ], h_step=[
        lambda reg: Activation('relu'),
    ],
    w_step=w_step,
)
model += Dense(num_classes, activation='softmax', trainable=False)
model.init(
    optimizer='adadelta', # TODO: adadelta needs to store the state; that is quite tricky, I think...
    loss='categorical_crossentropy',
    metrics=['categorical_accuracy']
)
# model._model.summary()

# Helper function for shuffle
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

batch_size = 512
p_x = []
p_y = {k: [] for k in model.get_depths().keys()}
history = []
for i in range(5000):
    print("Iteration {}".format(i))

    validate = i % 1 == 0

    if validate:
        validation_data = (x_test, y_test)
    else:
        validation_data = None

    res = model.train_batch(
        x_train, y_train, validation_data=validation_data,
        batch_size=batch_size, debug_print=True
    )

    if validate:
        history.append(res[-1])
        p_x.append(i)

        p = model._model.predict(x_test, batch_size=batch_size)
        total = len(y_test)
        ok = np.sum(np.argmax(y_test, axis=1) == np.argmax(p, axis=1))
        nok = total - ok
        print("test: {}/{} ok, p={}".format(ok, total, ok / total))

        p = model._model.predict(x_train, batch_size=batch_size)
        total = len(y_train)
        ok = np.sum(np.argmax(y_train, axis=1) == np.argmax(p, axis=1))
        nok = total - ok
        print("train: {}/{} ok, p={}".format(ok, total, ok / total))

    weights = model.get_depths()
    for p_k in weights.keys():
        p_y[p_k].append(weights[p_k])

    if i % 1 == 0:
        print("Create plots...")

        labels = sorted(history[0].keys())
        plt.figure(figsize=(20, 10))
        for k in labels:
            plt.plot(p_x, list(map(lambda e: e[k][0], history)))

        # labels = sorted(p_y.keys())
        # plt.figsize = (20,10)
        # for k in labels:
        #     plt.plot(p_x, p_y[k])

        plt.legend(labels)
        plt.grid(True)
        plt.savefig('data_01/w_{}_{}.png'.format(w_reg, f_reg))
        plt.clf()
        plt.close()
with open('data_01/w_{}_{}.pkl'.format(w_reg, f_reg), 'wb') as fh:
    pickle.dump(history, fh)
model._model.save('data_01/model.pkl')
