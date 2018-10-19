from typing import NamedTuple, Iterable, Callable, Any

import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.modules import Module
from torch.nn.parameter import Parameter

import numpy as np

import numbers

from ._deltaw import DeltaW_T, DeltaWBaseModule

def _get_object_fields_of_type(obj, cls) -> Iterable:
    return (m for m in inspect.getmembers(obj) if (not inspect.isroutine(m[1])) and isinstance(m[1], cls))


def _identity(x):
    return x


def freeze_layer(layer):
    for p in layer.parameters():
        p.requires_grad = False
    return layer

class GDeepModule(Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_deeper_base_layers(self) -> Iterable['GDeeperBase']:
        return _get_object_fields_of_type(self, GDeepModule)

    def _get_delta_w_layers(self) -> Iterable[DeltaWBaseModule]:
        return _get_object_fields_of_type(self, DeltaWBaseModule)

    def get_regularizations(self):
        return sum(l.get_regularizations() for l in self._get_delta_w_layers())

    def get_w_values(self, recursive=True):
        yield from ((l[0], l[1].get_w()) for l in self._get_delta_w_layers())
        if recursive:
            for name, layer in self._get_deeper_base_layers():
                yield from ((f'{name}.{l[0]}', l[1]) for l in layer.get_w_values(recursive))


class GDeepLayerBase(GDeepModule):
    class _LayerF(NamedTuple):
        f: Callable
        regularization: Any

    def __init__(
        self,
        f_i_builder, f_regularization = 0.,
        merge=lambda delta_w_i, x, x_new: x + delta_w_i * x_new,
        h_builder=lambda: _identity,
        w_regularization = 0., w_reguralizer_fn = lambda w: w.norm(1),
        f_delta_w_builder = DeltaW_T,
        *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self._f_i_builder = f_i_builder
        self._f_regularization = f_regularization

        self._merge = merge

        self._h_builder = h_builder

        self._w_regularization = w_regularization
        self._w_reguralizer_fn = w_reguralizer_fn
        self._f_delta_w = f_delta_w_builder()

        self._parameter_count = 0
        self._layers = []
        self._h = self._h_builder()

    def _register_parameter(self, parameter):
        name = f'{type(parameter).__name__}_param_{self._parameter_count}'
        self._parameter_count += 1
        self.register_parameter(name, parameter)

    def _register_parameters(self, parameters):
        def get_parameters():
            for parameter in parameters:
                if isinstance(parameter, Module):
                    yield from param.parameters()
                elif iter(parameters):
                    yield from parameters
                else:
                    yield parameter
        for parameter in get_parameters():
            self._register_parameter(parameter)

    def _build_layer(self, i):
        f, f_reg = self._f_i_builder(i, self._f_regularization, self._register_parameters)
        return GDeepLayerBase._LayerF(f=f, regularization=f_reg)

    def _get_layer(self, layer_index):
        while len(self._layers) <= layer_index:
            self._layers.append(self._build_layer(layer_index))
        return self._layers[layer_index]

    def forward(self, x):
        x0 = x
        def _forward_layers_rec(x_prev, layers):
            x_h = self._h(x_prev)
            if layers:
                layer_index = layers[0]
                x_new = self._get_layer(layer_index).f(x0, x_prev)
                x_merged = self._merge(self._f_delta_w(layer_index), x_prev, x_new)
                x_res = _forward_layers_rec(x_merged, layers[1:])
            else:
                x_res = x_h
            return x_res
        return _forward_layers_rec(x, list(self._f_delta_w.get_active_layers()))

    def get_w(self):
        return self._f_delta_w.get_w()

    def get_regularizations(self):
        reg = super().get_regularizations()

        # Get the w-regularization factor:
        # The value may be just a number or a function (that returns a number).
        w_regularization = self._w_regularization
        if not isinstance(w_regularization, numbers.Number):
            w_regularization = w_regularization()

        # Get the DeltaW regularization
        reg += w_regularization * self._w_regularizer_fn(self._delta_w.weight)

        # Add all other regularizations
        for i in self._delta_w.get_active_layers():
            layer = self._get_layer(i)
            f_reg = layer['f_reg']
            if f_reg is not None:
                reg += self._delta_w(i) * f_reg()

        return reg

class GDeepVLayer(GDeepLayerBase):
    def __init__(self, f_i_builder, *args, **kwargs):
        def vertical_f_i_builder(*args, **kwargs):
            f_i, f_i_reg = f_i_builder(*args, **kwargs)
            return (lambda x0, x_prev: f_i(x_prev)), f_i_reg
        super().__init__(f_i_builder=vertical_f_i_builder, *args, **kwargs)


class GDeepHLayer(GDeepLayerBase):
    def __init__(self, f_i_builder, *args, **kwargs):
        def horicontal_f_i_builder(*args, **kwargs):
            f_i, f_i_reg = f_i_builder(*args, **kwargs)
            return (lambda x0, x_prev: f_i(x0)), f_i_reg
        super().__init__(f_i_builder=horicontal_f_i_builder, *args, **kwargs)

