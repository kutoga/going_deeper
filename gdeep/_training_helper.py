import torch
from torch.autograd import Variable
from torch.utils.data.dataset import random_split
import torch.optim as optim
from lazy_load import lazy_func
import torch.nn.functional as F
import json

def split_dataset(train_data = None, validation_data = None, test_data = None, seed = None):
    if seed is not None:
        torch.manual_seed(seed)
    # Generate:
    # - Training data
    # - Validation data
    # - Training validation data
    # - Test data
    def _split_dataset(percentage_first, dataset):
        if not (0 <= percentage_first <= 1):
            raise ValueError()
        len_first = int(percentage_first * len(dataset))
        return random_split(dataset, [len_first, len(dataset) - len_first])

    if test_data is None:
        test_data, train_data = _split_dataset(0.2, train_data)
    if validation_data is None:
        validation_data, train_data = _split_dataset(0.2, train_data)
    train_validation_data, train_data = _split_dataset(0.1, train_data)
    return train_data, validation_data, train_validation_data, test_data

class History:
    def __init__(self):
        self._data = []
        self._epoch = 0
        self._batch = 0
        self._current_obj = None
        self._init_current_obj()

    def _init_current_obj(self):
        self._current_obj = {'batch': self._batch, 'epoch': self._epoch}

    def batch_finished(self):
        self._data.append(self._current_obj)
        self._init_current_obj()
        self._batch += 1

    def epoch_finished(self):
        self._epoch += 1

    def add_values(self, **values):
        self._current_obj.update(values)

    def get_newest(self, key, default = None, include_current = True):
        def get_objects():
            if include_current:
                yield self._current_obj
            yield from reversed(self._data)
        for obj in get_objects():
            if key in obj:
                return obj[key]
        return default

    def save(self, fh):
        json.dump(self._data, fh)

    def __getitem__(self, key):
        return self.get_newest(key)

class ModelTraining:
    def __init__(self, model, train_data, validation_data = None, test_data = None, seed = None, optimizer = None, *, batch_size = 32, shuffle = True, history_file = None):
        self._model = model
        self._history = History()
        train_data, validation_data, train_validation_data, test_data = split_dataset(train_data, validation_data, test_data, seed=seed)
        self._train_data = train_data
        self._validation_data = validation_data
        self._train_validation_data = train_validation_data
        self._test_data = test_data
        self._optimizer = optimizer or optim.Adamax(filter(lambda p: p.requires_grad, self._model.parameters()))
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._history_file = history_file

        self._dl_train = self._get_data_loader(self._train_data)
        self._dl_validation = self._get_data_loader(self._validation_data)
        self._dl_train_validation = self._get_data_loader(self._train_validation_data)
        self._dl_test = self._get_data_loader(self._test_data)

    @lazy_func
    def _get_data_loader(self, dataset, **kwargs):
        return torch.utils.data.DataLoader(dataset, batch_size=self._batch_size, shuffle=self._shuffle, **kwargs)

    def _train_batch(self, data, target):
        self._model.train()

        def _compute_accuracy(output, v_target):
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct = pred.eq(v_target.data.view_as(pred)).cpu().sum()
            return correct.item() / len(v_target)

        def _compute_loss(data, target):
            v_data = Variable(data)
            v_target = Variable(target)

            last_train_loss = self._history.get_newest('train_loss')
            last_train_valid_loss = self._history.get_newest('train_valid_loss')

            if None in (last_train_loss, last_train_valid_loss):
                last_train_loss = 1.
                last_train_valid_loss = 0.

            output = self._model(v_data)

            def relu(x):
                return max([0., x])

            d = abs(last_train_loss - last_train_valid_loss) / (1e-7 + abs(last_train_loss))

            delta_w_loss_weight = relu(last_train_loss - last_train_valid_loss)

            self._history.add_values(delta_w_loss_weight=delta_w_loss_weight)

            loss = F.nll_loss(output, v_target) + Variable(torch.FloatTensor([delta_w_loss_weight]).view([])) * self._model.get_regularizations()
            acc = _compute_accuracy(output, v_target)
            return loss, acc

        self._optimizer.zero_grad()
        train_loss, train_acc = _compute_loss(data, target)
        train_loss.backward()
        self._optimizer.step()

        # Evaluate the training validation data
        self._model.eval()
        train_valid_loss, train_valid_acc = _compute_loss(*next(iter(self._dl_train_validation)))

        self._history.add_values(
            train_loss = train_loss.item(),
            train_acc = train_acc,
            train_valid_loss = train_valid_loss.item(),
            train_valid_acc = train_valid_acc
        )

        print(f"Epoch:                          {self._history['epoch']}")
        print(f"Batch:                          {self._history['batch']}")
        print(f"Training Loss:                  {self._history['train_loss']}")
        print(f"Training Validation Loss:       {self._history['train_valid_loss']}")
        print(f"Training accuracy:              {self._history['train_acc']}")
        print(f"Training Validation accuracy:   {self._history['train_valid_acc']}")
        print(f"W-Values:                       {list(self._model.get_w_values())}")
        print(f"W-Loss Weight:                  {self._history['delta_w_loss_weight']}")
        print()

        self._history.batch_finished()

    def train_epoch(self):
        for batch_idx, (data, target) in enumerate(self._dl_train):
            self._train_batch(data, target)
        self._history.epoch_finished()
        if self._history_file is not None:
            with open(self._history_file, 'w') as fh:
                self._history.save(fh)

    def train(self, epochs = None):
        # TODO: Refactor (I could write this to any function in this project)
        if epochs is None:
            while True:
                self.train_epoch()
        else:
            for i in range(epochs):
                self.train_epoch()

    def test(self):
        pass

    def validate(self):
        pass

def train_epoch(model, ):
    # Create a simple output and log some data (maybe even create graphs?)
    pass
