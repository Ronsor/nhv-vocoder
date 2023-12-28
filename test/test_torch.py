#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright (c) 2021 Kazuhiro KOBAYASHI <root.4mac@gmail.com>
#
# Distributed under terms of the MIT license.

"""

"""


import matplotlib as mpl
import numpy.testing as npt
import torch
import torch.nn.functional as F

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa

B = 3


def test_interpolate():
    x = torch.randn((1, 10, 513))
    mask = x < 0
    x[mask] = -100


def test_ccep_calculation():
    window_size = 1024
    fft_size = window_size
    win = torch.hann_window(window_size)
    x = torch.randn(B, 1, window_size) * win

    # to fft->ifft
    y = torch.fft.fft(x, n=fft_size, dim=-1)
    x_ret = torch.fft.ifft(y, n=fft_size, dim=-1).real
    npt.assert_almost_equal(x.numpy(), x_ret.numpy(), decimal=5)

    # win -> logmag, phase -> ccep -> logmag, phase -> win
    log_mag, phase = torch.log(torch.abs(y)), torch.angle(y)
    comp = torch.complex(log_mag, phase)
    ccep = torch.fft.ifft(comp, n=fft_size, dim=-1)
    logmagphase = torch.fft.fft(ccep, n=fft_size, dim=-1)
    magphase = torch.exp(logmagphase)
    x_ret2 = torch.fft.ifft(magphase, n=fft_size, dim=-1).real
    npt.assert_almost_equal(x.numpy(), x_ret2.numpy(), decimal=5)


def test_conv1d():
    x = torch.randn(1, 1, 1024 + 256 * 2 - 1)
    k = torch.randn(1, 2, 1024)
    _ = F.conv1d(x[..., :-256], k[:, :1])
    _ = F.conv1d(x[..., 256:], k[:, 1:])
