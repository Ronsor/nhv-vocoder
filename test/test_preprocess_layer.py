#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright (c) 2021 Kazuhiro KOBAYASHI <root.4mac@gmail.com>
#
# Distributed under terms of the MIT license.

"""

"""

from pathlib import Path

import matplotlib as mpl
import soundfile as sf
import torch

mpl.use("agg")
import matplotlib.pyplot as plt  # noqa

dirname = Path(__file__).parent
wavf = dirname / "data" / "test.wav"


def test_melspectrogram():
    from nhv.layer import LogMelSpectrogram

    x, fs = sf.read(wavf)

    hop_size = 128
    fft_size = 1024
    win_length = 1024
    window = "hann"
    n_mels = 80
    fmin = None
    fmax = None
    mel_layer = LogMelSpectrogram(
        fs=fs,
        hop_size=hop_size,
        fft_size=fft_size,
        win_length=win_length,
        window=window,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        center=True,
        pad_mode="reflect",
    )
    x = torch.from_numpy(x).unsqueeze(0).float()
    x = mel_layer(x)


def test_melspec2logspec():
    from nhv.layer import LogMelSpectrogram2LogMagnitude  # noqa
    from nhv.layer import CepstrumLiftering, LogMelSpectrogram, Magnitude

    x, fs = sf.read(wavf)
    x = torch.from_numpy(x).unsqueeze(0).float()

    hop_size = 128
    fft_size = 1024
    win_length = 1024
    window = "hann"
    n_mels = 80
    fmin = 80
    fmax = 7600

    mag_layer = Magnitude(
        fs=fs,
        hop_size=hop_size,
        fft_size=fft_size,
        win_length=win_length,
        window=window,
    )
    mag = mag_layer(x)
    log_mag = torch.clamp(mag, min=1e-10).log10()

    log_mel_layer = LogMelSpectrogram(
        fs=fs,
        hop_size=hop_size,
        fft_size=fft_size,
        win_length=win_length,
        window=window,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        center=True,
        pad_mode="reflect",
    )
    inv_mel_layer = LogMelSpectrogram2LogMagnitude(
        fs=fs,
        fft_size=fft_size,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        eps=1.0e-10,
        roll_size=16,
    )
    log_melspc = log_mel_layer(x)
    log_mag_inv = inv_mel_layer(log_melspc)

    cep_layer = CepstrumLiftering(lifter_size=24)
    log_mag_inv_lifter = cep_layer(log_mag_inv)

    T, D = 100, 512
    plt.figure()
    plt.plot(log_mag_inv.squeeze().cpu().numpy()[T, :D])
    plt.plot(log_mag.squeeze().cpu().numpy()[T, :D])
    plt.plot(log_mag_inv_lifter.squeeze().cpu().numpy()[T, :D])
    plt.savefig(dirname / "data" / "test_melspec2logspec.png")

    plt.figure()
    plt.plot(log_melspc.squeeze().cpu().numpy()[T])
    plt.savefig(dirname / "data" / "test_log_melspc.png")
