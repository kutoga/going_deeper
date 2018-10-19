import torch
import torch.nn.functional as F

from torch.nn.modules import Module
from torch.nn.parameter import Parameter

import numpy as np

 # TODO: Refactor complete file

def integers_in_range(x_start, x_end, include_borders=True):
    x0 = int(np.ceil(x_start))
    x1 = int(np.floor(x_end))
    if x0 == x_start and include_borders:
        res_start = x0
    else:
        res_start = x0 + 1
    if x1 == x_end and include_borders:
        res_end = x1
    else:
        res_end = x1 + 1
    return range(res_start, res_end + 1)


class DeltaWBaseModule(Module):
    def __init__(self):
        super().__init__()

    def get_w(self):
        pass

class DeltaW_E(DeltaWBaseModule):
    """
    A DeltaW function based on exp; this is the most trivial implementation. It is continuous, but
    the derivative is not.

    The used function is:

    f(x) = ReLU(1 - exp((x - w) / s))

    with a trainable weight w and a positive constant s (stretch-factor)
    (important: the property to have f(w)=0 must still hold, therefore not x itself is stretched)
    """
    def __init__(self, init=1.0, stretch=1.0):
        super().__init__()
        self.weight = Parameter(torch.Tensor((1,)).fill_(init))
        self.stretch = stretch

    def forward(self, x):
        return F.relu(1 - torch.exp((x - self.weight) / self.stretch))

    def get_active_layers(self):
        w = self.get_w()
        layers = np.maximum(0, int(np.ceil(w)))
        return range(layers)

    def get_w(self):
        return self.weight[0].item()

    def __repr__(self):
        return self.__class__.__name__


class DeltaW_T(DeltaWBaseModule):
    """
    An improved version of the DeltaW function. It is everywhere continuous and the same is true for its
    derivative.

    The used function is:

    f(x) = tanh(ReLU(-(x - w) / s)^2)

    with a trainable weight w and a positive constant s (stretch-factor)
    (important: the property to have f(w)=0 must still hold, therefore not x itself is stretched)
    """
    def __init__(self, init=1.0, stretch=1.0):
        super().__init__()
        self.weight = Parameter(torch.Tensor((1,)).fill_(init))
        self.stretch = stretch

    def forward(self, x):
        rx = F.relu(-(x - self.weight) / self.stretch)
        return torch.tanh(rx * rx)

    def get_active_layers(self):
        w = self.get_w()
        layers = np.maximum(0, int(np.ceil(w)))
        return range(layers)

    def get_w(self):
        return self.weight[0].item()

    def __repr__(self):
        return self.__class__.__name__


class DeltaW_R(DeltaWBaseModule):
    """
    A simplified version of the DeltaW function. It is everywhere continuous, but its derivative is not. The calculation
    of this function is very simple (compared to DeltaW_E and DeltaW_T).

    The used function is:

    f(x) = max(0, min(1, k*(-(x - w))))

    For a fixed k that is usually <1 (e.g. 0.5)

    with a trainable weight w
    """
    def __init__(self, init=1.0, k=0.5):
        super().__init__()
        self.weight = Parameter(torch.Tensor((1,)).fill_(init))
        self.k = k

    def forward(self, x):
        return (self.k * (- (x - self.weight))).clamp(0, 1)

    def get_active_layers(self):
        w = self.get_w()
        layers = np.maximum(0, int(np.ceil(w)))
        return range(layers)

    def get_w(self):
        return self.weight[0].item()

    def __repr__(self):
        return self.__class__.__name__


class DeltaW_S(DeltaWBaseModule):
    """
    A special DeltaW-function that works as a selector. It only enables a limited range of values. The function and
    its derivative are everywhere continuous.

    The function is defined in the following way (g(x) and h(x) are helper functions):

    g(x) = (cos(x) + 1) / 2
    h(x) = g(min(-pi, max(pi, x)))

    f(x) = h(x * 2 * pi / r)

    The variable r defines the length (range) of the selection. This value be larger than 2, otherwise the training might
    stock.


    Obsolete (previous idea):
    The function is defined in the following way (g(x) and h(x) are helper functions):

    g(x) = (sec(x)*sech(tan(x)))^2
    h(x) = g(min(pi/2, max(-pi/2, x)))

    f(x) = h(x * pi / r)

    The variable r defines the length (range) of the selection. This value should be larger than 2, otherwise the training might
    stock.

    """
    def __init__(self, init=1.0, r=np.pi):
        super().__init__()
        self.weight = Parameter(torch.Tensor((1,)).fill_(init))
        self.r = r

    def forward(self, x):
        def g(x):
            return (torch.cos(x) + 1.) / 2.
        def h(x):
            return g(x.clamp(-np.pi, np.pi))
        def f(x):
            return h(x * 2. * np.pi / self.r)
        return f(x)

    def get_active_layers(self):
        # Layers are active if they are inside (excluding the border): (w-r/2, w+r/2)
        w = self.get_w()
        return integers_in_range(w - self.r / 2, w + self.r / 2, include_borders=False)

    def get_w(self):
        return self.weight[0].item()

    def __repr__(self):
        return self.__class__.__name__


class DeltaW_SR(DeltaWBaseModule):
    """
    A special DeltaW-function that works as a selector. It only enables a limited range of values. The function is
    everywhere continuous, but its derivative is not.

    It is basically a simplified version of DeltaW_S.

    The used formula is:

    f(x) = ReLU(1 - abs((x - w) * 2 / r))

    The variable r defines the length (range) of the selection. This value should be larger than 2, otherwise the training might
    stock.

    """
    def __init__(self, init=1.0, r=np.pi):
        super().__init__()
        self.weight = Parameter(torch.Tensor((1,)).fill_(init))
        self.r = r

    def forward(self, x):
        return F.relu(1 - torch.abs((x - self.weight) * 2 / self.r))

    def get_active_layers(self):
        # Layers are active if they are inside (excluding the border): (w-r/2, w+r/2)
        w = self.get_w()
        return integers_in_range(w - self.r / 2, w + self.r / 2, include_borders=False)

    def get_w(self):
        return self.weight[0].item()

    def __repr__(self):
        return self.__class__.__name__

