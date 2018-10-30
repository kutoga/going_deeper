
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data.dataset import random_split

from torchvision import datasets, transforms

from itertools import islice

from gdeep import *

cuda = False

# TODO: Make this shit working
# delta_w_f = lambda *args, **kwargs: w_dependend_stretch(DeltaW_E(*args, **kwargs), lambda w: w^0.5)
delta_w_f = DeltaW_E
delta_w_f = lambda: DeltaW_T(stretch=10)

dataset_name = 'CIFAR10'

w_regularization = 1e-9
f_regularization = 1e-2

use_cuda = torch.cuda.is_available()

class TestCNN(GDeepModule):
    def __init__(self, cnn_l0_features = 8, cnn_l1_features = 16, deltaw_builder=DeltaW_T, **kwargs):
        super().__init__(**kwargs)
        self._cnn_l0_features = cnn_l0_features
        self._cnn_l1_features = cnn_l1_features

        self._l0_cnn = torch.nn.Conv2d(3 if dataset_name == 'CIFAR10' else 1, cnn_l0_features, (3, 3), stride=1, padding=1)
        self._l0_act = torch.nn.ReLU()
        self._l0_bn = torch.nn.BatchNorm2d(cnn_l0_features)

        def f_i_l1_builder(i, reg, add_params):
            li_cnn = torch.nn.Conv2d(cnn_l0_features, cnn_l0_features, (3, 3), stride=1, padding=1)
            li_bn = torch.nn.BatchNorm2d(cnn_l0_features)
            li_act = torch.nn.ReLU()

            add_params(li_cnn, li_act, li_bn)

            if use_cuda:
                li_cnn.cuda()
                li_bn.cuda()
                li_act.cuda()

            def f_i(x):
                x = li_cnn(x)
                x = li_act(x)
                x = li_bn(x)
                return x
                return fluent_compose(
                    li_cnn, li_act, li_bn
                )(x)

            def f_i_reg():
                return reg * (li_bn.bias.norm(2) + li_bn.weight.norm(2))

            return f_i, f_i_reg

        self._l1 = GDeepVLayer(f_i_l1_builder, f_regularization=f_regularization, w_regularization=w_regularization, f_delta_w_builder=delta_w_f)
        self._l2_max = torch.nn.MaxPool2d((2, 2), stride=(2, 2))
        self._l3_cnn = torch.nn.Conv2d(cnn_l0_features, cnn_l1_features, (3, 3), stride=1, padding=1)
        self._l3_act = torch.nn.ReLU()
        self._l3_bn = torch.nn.BatchNorm2d(cnn_l1_features)

        def f_i_l4_builder(i, reg, add_params):
            li_cnn = torch.nn.Conv2d(cnn_l1_features, cnn_l1_features, (3, 3), stride=1, padding=1)
            li_bn = torch.nn.BatchNorm2d(cnn_l1_features)
            li_act = torch.nn.ReLU()

            add_params(li_cnn, li_act, li_bn)

            if use_cuda:
                li_cnn.cuda()
                li_bn.cuda()
                li_act.cuda()

            def f_i(x):
                x = li_cnn(x)
                x = li_act(x)
                x = li_bn(x)
                return x

            def f_i_reg():
                return reg * (li_bn.bias.norm(2) + li_bn.weight.norm(2))

            return f_i, f_i_reg

        self._l4 = GDeepHLayer(f_i_l4_builder, f_regularization=f_regularization, w_regularization=w_regularization, f_delta_w_builder=delta_w_f)
        self._l5_max = torch.nn.MaxPool2d((2, 2), stride=(2, 2))
        if dataset_name == 'CIFAR10':
            self._l6_fc = torch.nn.Linear(8 * 8 * cnn_l1_features, 10)
        else:
            self._l6_fc = torch.nn.Linear(7 * 7 * cnn_l1_features, 10)

    def forward(self, x):
        x = self._l0_cnn(x)
        x = self._l0_act(x)
        x = self._l0_bn(x)

        x = self._l1(x)

        x = self._l2_max(x)

        x = self._l3_cnn(x)
        x = self._l3_act(x)
        x = self._l3_bn(x)

        x = self._l4(x)

        x = self._l5_max(x)

        if dataset_name == 'CIFAR10':
            x = x.view(-1, 8 * 8 * self._cnn_l1_features)
        else:
            x = x.view(-1, 7 * 7 * self._cnn_l1_features)

        x = self._l6_fc(x)

        x = F.log_softmax(x, dim=1)
        return x

        return fluent_compose(
            self._l0_cnn,
            self._l0_act,
            self._l0_bn,
            self._l1,
            self._l2_max,
            self._l3_cnn,
            self._l3_act,
            self._l3_bn,
            self._l4,
            self._l5_max,
            self._l6_fc
        )(x)

kwargs = {}

batch_size = 128
shuffle = True
data_dir = '.tmp/data'

if dataset_name == 'MNIST':
    # MNIST
    ds = datasets.MNIST
    org_train_data = ds(data_dir, train=True, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]))
    org_test_data = ds(data_dir, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]))

elif dataset_name == 'CIFAR10':
    # CIFAR10
    ds = datasets.CIFAR10
    org_train_data = ds(data_dir, train=True, download=True,
                        transform=transforms.Compose([
                            transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                        ]))
    org_test_data = ds(data_dir, train=False, transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                        ]))
else:
    raise RuntimeError(f"Invalid dataset: {dataset_name}")

# TODO: pass cuda setting somehow to the network / training
tr = ModelTraining(TestCNN(), train_data=org_train_data, test_data=org_test_data, batch_size=64, seed=1729, history_file=f'.tmp/history.json')
tr.train()
