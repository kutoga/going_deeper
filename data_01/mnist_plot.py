from os import path
import glob
import pickle
import matplotlib.pyplot as plt

input_dir = '.'

pkl_files = glob.glob(path.join(input_dir, 'w*.pkl'))

# Load all files with their content
def load(pkl_file):
    with open(pkl_file, 'rb') as fh:
        return pickle.load(fh)
data = list(map(
    lambda f: {
        'path': f,
        'data': load(f)
    },
    sorted(pkl_files)
))

# Create the x values
x = list(range(1, len(data[0]['data']) + 1))

# Get all keys
keys = sorted(data[0]['data'][0].keys())

# Define a function to get a field for each timestep of a pkl-file
def get_data(index, field):
    return list(map(
        lambda x: x[field][0],
        data[index]['data']
    ))


def save_plot(columns, y_label, output_file, block=False):

    # Create plots
    plt.figure(1, (12, 5))

    for column in columns:
        # plt.plot(x, get_data(i, 'valid_loss'))
        plt.plot(x, get_data(0, column[0]))

    plt.legend(list(map(lambda d: d['path'], data)))

    plt.legend(list(map(lambda c: c[1], columns)), loc='upper right')

    # plt.legend(list(reversed(list(map(lambda i: '{} active inputs'.format(i), range(len(data)))))), loc='upper right')

    plt.grid(True)
    plt.xlabel('iteration')
    plt.ylabel(y_label)

    plt.show(block=block)
    plt.savefig(output_file)
    plt.clf()
    plt.close()

save_plot([
    ('d:dcnn0', 'cnn0_w'),
    ('d:dcnn1', 'cnn1_w'),
    ('d:dfc0', 'fc0_w'),
], 'weight', 'mnist_weights.png')

save_plot([
    ('val_categorical_accuracy', 'test accuracy'),
], 'accuracy', 'mnist_accuracy.png')

save_plot([
    ('loss', 'training loss'),
    ('val_loss', 'test loss'),
], 'loss', 'mnist_loss.png')

# save_plot('d:d0', 'w', 'xor_weights.png')
# save_plot('val_loss', 'validation loss', 'xor_test_loss.png')
# save_plot('val_acc', 'validation accuracy', 'xor_test_accuracy.png')

pass
