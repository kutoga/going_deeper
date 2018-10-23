
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

f_regularization = 1e-8
w_regularization = 1e-8

# TODO: Make this shit working
# delta_w_f = lambda *args, **kwargs: w_dependend_stretch(DeltaW_E(*args, **kwargs), lambda w: w^0.5)
delta_w_f = DeltaW_E

class TestCNN(GDeepModule):
    def __init__(self, cnn_l0_features = 8, cnn_l1_features = 16, deltaw_builder=DeltaW_T):
        super().__init__()
        self._cnn_l0_features = cnn_l0_features
        self._cnn_l1_features = cnn_l1_features

        self._l0_cnn = torch.nn.Conv2d(1, cnn_l0_features, (3, 3), stride=1, padding=1)
        self._l0_act = torch.nn.ReLU()
        self._l0_bn = torch.nn.BatchNorm2d(cnn_l0_features)

        def f_i_l1_builder(i, reg, add_params):
            li_cnn = torch.nn.Conv2d(cnn_l0_features, cnn_l0_features, (3, 3), stride=1, padding=1)
            li_bn = torch.nn.BatchNorm2d(cnn_l0_features)
            li_act = torch.nn.ReLU()

            add_params(li_cnn, li_act, li_bn)

            def f_i(x):
                x = li_cnn(x)
                x = li_act(x)
                x = li_bn(x)
                return x
                return fluent_compose(
                    li_cnn, li_bn, li_act
                )(x)

            def f_i_reg():
                return reg * li_bn.weight.norm(2)

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

            def f_i(x):
                x = li_cnn(x)
                x = li_act(x)
                x = li_bn(x)
                return x

            def f_i_reg():
                return reg * li_bn.weight.norm(2)

            return f_i, f_i_reg

        self._l4 = GDeepHLayer(f_i_l4_builder, f_regularization=f_regularization, w_regularization=w_regularization, f_delta_w_builder=delta_w_f)
        self._l5_max = torch.nn.MaxPool2d((2, 2), stride=(2, 2))
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

ds = datasets.MNIST
kwargs = {}

batch_size = 64
shuffle = True

org_train_data = ds('../data', train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]))
org_test_data = ds('../data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]))

# Generate the required datasets
torch.manual_seed(1729)
ds_train_len = int(0.9 * len(org_train_data))
ds_train, ds_train_valid = random_split(org_train_data, [ds_train_len, len(org_train_data) - ds_train_len])
ds_test = org_test_data

train_loader = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=shuffle, **kwargs)
train_valid_loader = torch.utils.data.DataLoader(ds_train_valid, batch_size=batch_size // 4, shuffle=shuffle, **kwargs)
test_loader = torch.utils.data.DataLoader(ds_test, batch_size=batch_size, shuffle=shuffle, **kwargs)

model = TestCNN()
optimizer = optim.Adamax(filter(lambda p: p.requires_grad, model.parameters()))
criterion = nn.BCELoss()

model.train()

_last_train_loss = 1.
_last_train_valid_loss = 0.

for batch_idx, (data, target) in enumerate(train_loader):
    model.train()
    data, target = Variable(data), Variable(target)
    optimizer.zero_grad()
    output = model(data)
    loss = F.nll_loss(output, target) + F.relu(Variable(torch.FloatTensor([_last_train_loss - _last_train_valid_loss]).view([]))) * model.get_regularizations()
    loss.backward()
    optimizer.step()
    train_loss = loss.item()
    print(f"Batch: {batch_idx}")
    print(f"Loss: {loss.item()}")
    print(f"w-values: {list(model.get_w_values())}")

    # Train-Valid loss
    model.eval()
    data, target = list(islice(train_valid_loader, 1))[0]
    data, target = Variable(data), Variable(target)
    output = model(data)
    #TODO: Relative measure instead of (absolute): "_last_train_loss - _last_train_valid_loss"
    loss = F.nll_loss(output, target) + F.relu(Variable(torch.FloatTensor([_last_train_loss - _last_train_valid_loss]).view([]))) * model.get_regularizations()
    train_valid_loss = loss.item()

    _last_train_loss = train_loss
    _last_train_valid_loss = train_valid_loss
    print(f"_last_train_loss=       {_last_train_loss}")
    print(f"_last_train_valid_loss= {_last_train_valid_loss}")
    print()