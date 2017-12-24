import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import numpy as np

from keras.layers import Input, Dense, Activation, Dropout

from ginfty import TDModel, GInftlyLayer, GammaRegularizedBatchNorm, c_l2

# Configuration
n_input_units = 8
n_used_input_units = 10
n_internal_units = 24

f_reg = 1e-8
w_reg = 1e-4

# Do a test for 8, 7, ... 0 active inputs
for n_used_input_units in reversed(range(0, n_input_units + 1)):

    assert 0 <= n_used_input_units <= n_input_units

    # Create the model
    model = TDModel()
    model += Input((n_input_units,))
    model += Dense(n_internal_units, activation='relu', trainable=False)
    model += GInftlyLayer(
        'd0',
        f_layer=[
            lambda reg: Dense(n_internal_units),
            lambda reg: GammaRegularizedBatchNorm(reg=reg, max_free_gamma=0.),
            lambda reg: Dropout(0.1),
        ], h_step=[
            lambda reg: Activation('relu')
        ],
        w_regularizer=(c_l2, w_reg),
        f_regularizer=(c_l2, f_reg)#1e-2)
    )
    model += Dense(1, activation='sigmoid', trainable=False)

    # Create some helper functions
    def generate_xor_data(n):
        x = np.random.uniform(0, 1, (n, n_input_units))
        x[x>=.5] = 1.
        x[x<.5] = 0.
        y = np.sum(x[:, :n_used_input_units], axis=1) % 2
        y[y > 1] = 0.
        return x, y

    # Build and train the model
    model.init(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    batch_size = 1000
    p_x = []
    p_y = {k:[] for k in model.get_depths().keys()}
    history = []
    for i in range(100000):
        print("Iteration {}".format(i))
        x, y = generate_xor_data(batch_size)

        validate = i % 100 == 0

        if validate:
            validation_data = generate_xor_data(batch_size * 10)
        else:
            validation_data = None

        res = model.train_step(x, y, validation_data, debug_print=False)

        if validate:
            history.append(res)
            p_x.append(i)

        weights = model.get_depths()
        for p_k in weights.keys():
            p_y[p_k].append(weights[p_k])

        if i % 1000 == 0:

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
            plt.savefig('data_00/w_{}_{}_{}_{}.png'.format(n_input_units, n_used_input_units, n_internal_units, f_reg))
            plt.clf()
            plt.close()
    with open('data_00/w_{}_{}_{}_{}.pkl'.format(n_input_units, n_used_input_units, n_internal_units, f_reg), 'wb') as fh:
        pickle.dump(history, fh)
