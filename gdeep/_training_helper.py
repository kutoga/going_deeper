import torch
from torch.autograd import Variable
from torch.utils.data.dataset import random_split
from lazy_load import lazy_func

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
        len_first = int(percentage_first * dataset)
        return random_split(dataset, [len_first, len(dataset) - len_first])

    if test_data is None:
        test_data, train_data = _split_dataset(0.2, train_data)
    if validation_data is None:
        validation_data, train_data = _split_dataset(0.2, train_data)
    train_validation_data = _split_dataset(0.1, train_data)
    return train_data, validation_data, train_validation_data, test_data

class History:
    def __init__(self):
        self._data = []
        self._batch
        self._current_obj = None
        self._init_current_obj()

    def _init_current_obj(self):
        self._current_obj = {'batch': len(self._data)}

    def batch_finished(self):
        self._data.append(self._current_obj)
        self._init_current_obj()

    def add_values(**values):
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

class ModelTraining:
    def __init__(self, model, train_data, validation_data, test_data, seed = None, optimizer = None, *, batch_size = 32, shuffle = True):
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

        self._dl_train = self._get_data_loader(self._train_data)
        self._dl_validation = self._get_data_loader(self._validation_data)
        self._dl_train_validation = self._get_data_loader(self._train_validation_data)
        self._dl_test = self._get_data_loader(self._test_data)

    @lazy_func
    def _get_data_loader(self, dataset, **kwargs):
        return torch.utils.data.DataLoader(dataset, batch_size=self._batch_size, shuffle=self._shuffle, **kwargs)

    def _train_batch(self, data, target):
        self._model.train()

        def _compute_loss(data, target):
            v_data = Variable(data)
            v_target = Variable(target)

            last_train_loss = self._history.get_newest('train_loss')
            last_train_valid_loss = self._history.get_newest('train_valid_loss')

            if None in (last_train_loss, last_train_valid_loss):
                last_train_loss = 1.
                last_train_valid_loss = 0.

            output = self._model(v_data)
            loss = F.nll_loss(output, v_target) + F.relu(Variable(torch.FloatTensor([last_train_loss - last_train_valid_loss]).view([]))) * model.get_regularizations()
            return loss

        self._optimizer.zero_grad()
        train_loss = _compute_loss(data, target)
        train_loss.backward()
        self._optimizer.step()

        # Evaluate the training validation data
        model.eval()
        train_valid_loss = _compute_loss(*next(iter(self._dl_train_validation)))

        self._history.add_values(
            'train_loss' = train_loss.item(),
            'train_valid_loss' = train_valid_loss.item()
        )
        self._history.batch_finished()

        ''' train_loss = loss.item()
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
        print() '''

    def train_epoch(self):
        for batch_idx, (data, target) in enumerate(self._dl_train):
            self._train_batch(data, target)

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
