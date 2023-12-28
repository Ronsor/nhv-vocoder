#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright (c) 2021 Kazuhiro KOBAYASHI <root.4mac@gmail.com>
#
# Distributed under terms of the MIT license.

"""

"""

import time
from pathlib import Path

import onnxruntime as ort
import torch
import torch.nn as nn
from nhv.layer import ConvLayers

datadir = Path(__file__).parent / "data"
onnxf = datadir / "test.onnx"

T = 54


class ParentNet(nn.Module):
    def __init__(self, input_size=80):
        super().__init__()
        # self.parent_conv1d = nn.Conv1d(input_size, input_size, 3, padding=1)
        self.child_net = ConvLayers(
            out_channels=input_size,
            n_conv_layers=5,
            conv_type="ddsconv",
            use_causal=True,
        )
        self.child_net2 = ConvLayers(
            out_channels=input_size,
            n_conv_layers=5,
            conv_type="ddsconv",
            use_causal=True,
        )

        # NOTE: it remains hook handles to remove old hooks
        self.handles = []

    def _forward_without_cache(self, x):
        x1 = self.child_net(x)
        x2 = self.child_net2(x)
        return x1 + x2

    def forward(self, caches, *args):
        self.caches = caches
        self.new_caches = []
        self.cache_num = 0
        x = self._forward_without_cache(*args)
        return x, self.new_caches

    def reset_cache(self, x):
        self.caches = []
        self.receptive_sizes = []
        self._initialize_caches(batch_size=x.size(0))

        # set ordering hook
        self._set_pre_hooks(cache_ordering=True)
        # caclulate order of inference
        _ = self._forward_without_cache(x)
        # remove hook handles for ordering
        [h.remove() for h in self.handles]
        # set concatenate hook
        self._set_pre_hooks(cache_ordering=False)
        return self.caches

    def _initialize_caches(self, batch_size=1):
        self.caches_dict = {}
        self.receptive_sizes_dict = {}
        for k, m in self.named_modules():
            if isinstance(m, nn.Conv1d):
                if m.kernel_size[0] > 1:
                    receptive_size = self._get_receptive_size_1d(m)
                    self.caches_dict[id(m)] = torch.randn(
                        (batch_size, m.in_channels, receptive_size)
                    )
                    self.receptive_sizes_dict[id(m)] = receptive_size

    def _set_pre_hooks(self, cache_ordering=True):
        if cache_ordering:
            func = self._cache_ordering
        else:
            func = self._concat_cache
        for k, m in self.named_modules():
            if isinstance(m, nn.Conv1d):
                if m.kernel_size[0] > 1:
                    self.handles.append(m.register_forward_pre_hook(func))

    def _concat_cache(self, module, inputs):
        def __concat_cache(inputs, cache, receptive_size):
            inputs = torch.cat([cache, inputs[0]], axis=-1)
            inputs = inputs[..., -receptive_size:]
            return inputs

        cache = self.caches[self.cache_num]
        receptive_size = self.receptive_sizes[self.cache_num]
        inputs = __concat_cache(inputs, cache, receptive_size)
        self.new_caches += [inputs]
        self.cache_num += 1
        return inputs

    def _cache_ordering(self, module, inputs):
        self.caches.append(self.caches_dict[id(module)])
        self.receptive_sizes.append(self.receptive_sizes_dict[id(module)])

    @staticmethod
    def _get_receptive_size_1d(m):
        return (m.kernel_size[0] - 1) * m.dilation[0] + 1


class ChildNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.child_conv1d_1 = nn.Conv1d(10, 10, 5, padding=2)
        self.child_conv1d_2 = nn.Conv1d(10, 10, 15, padding=7)

    def forward(self, x):
        x = self.child_conv1d_1(x)
        x = self.child_conv1d_2(x)
        return x


def test_torch_class():
    def _remove_padding(m):
        if isinstance(m, torch.nn.Conv1d):
            m.padding = (0,)
        if isinstance(m, torch.nn.Conv2d):
            m.padding = (0, 0)

    net = ParentNet()
    x = torch.randn(1, 140, 80)

    # create cache
    caches = net.reset_cache(x)
    net.apply(_remove_padding)

    x_frame = torch.randn(1, 1, 80)
    y, new_caches = net.forward(caches, x_frame)

    input_names = []
    input_names += [f"cache_{i}" for i in range(len(caches))]
    input_names += ["x"]
    output_names = ["y", "new_caches"]
    torch.onnx.export(
        net,
        (caches, x_frame),
        onnxf,
        verbose=False,
        input_names=input_names,
        output_names=output_names,
        opset_version=12,
    )

    # for c in caches:
    #     print(c.size())

    sess = ort.InferenceSession(str(onnxf))
    inputs = {
        "x": x_frame.cpu().numpy(),
    }
    for i, c in enumerate(caches):
        inputs.update({f"cache_{i}": c.cpu().numpy()})

    st = time.time()
    for t in range(T):
        _ = sess.run(None, inputs)
        inputs.update({"x": torch.randn(1, 1, 80).cpu().numpy()})
    print(f"Elapsed time: {(time.time() - st) / T}")
