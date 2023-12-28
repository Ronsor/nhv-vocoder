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

import h5py
import numpy as np
import numpy.testing as npt
import onnxruntime as ort
import pytest
import soundfile as sf
import torch
import torch.nn.functional as F
from nhv import IncrementalNeuralHomomorphicVocoder, NeuralHomomorphicVocoder
from nhv.layer import CCepLTVFilter, ConvLayers, SinusoidsGenerator
from nhv.layer.incremental import (IncrementalConvLayers,
                                   IncrementalSinusoidsGenerator)

B, T, D = 1, 54, 80
hop_size, fft_size = 128, 1024
# B, T, D = 1, 2, 2
# hop_size, fft_size = 2, 8
noise_std = 0.03
datadir = Path(__file__).parent / "data"
onnxf = datadir / "nhv.onnx"
checkpoint = datadir / "test_checkpoint.pkl"
scalerf = datadir / "test_scaler.pkl"
h5f = datadir / "test.h5"


def test_incremental_ccepltvfilter():
    fft_size = 4
    hop_size = 2
    window_size = hop_size * 2
    ltv_params = {
        "in_channels": 80,
        "n_ltv_layers": 3,
        "hop_size": hop_size,
        "fft_size": fft_size,
        "n_ltv_postfilter_layers": 0,
        "conv_type": "ddsconv",
        "use_causal": True,
        "ltv_postfilter_type": None,
    }
    base_net = CCepLTVFilter(**ltv_params)
    # base_net.remove_weight_norm()
    base_net.eval()

    # inference by base_net
    T = 3
    # z = torch.randn(B, 1, T * hop_size)
    # y = torch.randn((B, T, fft_size))
    z = torch.arange(B * T * hop_size).float().reshape(B, 1, T * hop_size)
    y = torch.ones((B, T, fft_size + 1))
    z_base = base_net._conv_impulse(z, y)
    base_out = base_net._ola(z_base)

    # inference by inc_net
    inc_out = []
    out_old = torch.zeros(1, 1, hop_size * 2)
    z = F.pad(z, (hop_size, 0))
    for t in range(T):
        p = t * hop_size
        z_t = z[..., p : p + window_size]
        z_t = F.pad(z_t, (fft_size // 2, fft_size // 2))
        y_t = y[:, t : t + 1]
        out = F.conv1d(z_t, y_t) * base_net.win
        sig = out_old[..., -hop_size:] + out[..., :hop_size]
        inc_out += [sig]
        out_old = out
    inc_out = torch.cat(inc_out, axis=-1).reshape(-1)
    base_out = base_out.reshape(-1)
    npt.assert_array_almost_equal(
        inc_out[..., hop_size:].squeeze().detach().numpy(),
        base_out[..., hop_size:].squeeze().detach().numpy(),
        decimal=5,
    )


@pytest.mark.parametrize("conv_type", ["original", "ddsconv"])
def test_incremental_conv_layers(conv_type):
    conv_params = {"use_causal": True, "conv_type": conv_type, "n_conv_layers": 3}
    base_net = ConvLayers(**conv_params)
    base_net.remove_weight_norm()
    base_net.eval()
    inc_net = IncrementalConvLayers(**conv_params)
    inc_net.remove_weight_norm()
    inc_net.load_state_dict(base_net.state_dict(), strict=True)
    inc_net.eval()
    x = torch.randn((B, T, D))
    base_out = base_net(x)

    conv_caches = inc_net.reset_caches(x, hop_size=hop_size, batch_size=B)
    conv_caches = [torch.zeros_like(c) for c in conv_caches]
    inc_out = []
    for n in range(T):
        y, conv_caches = inc_net.forward(
            x[:, n : n + 1],
            conv_caches,
        )
        inc_out += [y]
    inc_out = torch.cat(inc_out, axis=1)
    npt.assert_array_almost_equal(
        base_out.detach().numpy(), inc_out.detach().numpy(), decimal=5
    )


def test_incremental_impulse_generation():
    base_net = SinusoidsGenerator(hop_size=hop_size, fs=24000)
    inc_net = IncrementalSinusoidsGenerator(hop_size=hop_size, fs=24000)
    f0 = torch.arange(200, 200 + B * T, 1).reshape(B, T, 1).float()
    uv = torch.ones(B, T, 1)
    base_excit = base_net(f0, uv)

    cache = torch.zeros(1, 1, 1)
    inc_excit = []
    for t in range(T):
        e, cache = inc_net.incremental_forward(
            f0[:, t].unsqueeze(-1), uv[:, t].unsqueeze(-1), cache=cache
        )
        inc_excit += [e]
    inc_excit = torch.cat(inc_excit, dim=-1)
    print(base_excit, inc_excit)

    # NOTE: desimal > 3 causes error due to accumulation of calculation errors
    print(torch.topk(torch.abs(base_excit - inc_excit), 20))
    npt.assert_array_almost_equal(base_excit.numpy(), inc_excit.numpy(), decimal=2)


def get_nhv_params(
    conv_type="ddsconv", postfilter_type=None, ltv_postfilter_type=None, checkpoint=None
):
    nhv_params = {
        "fs": 24000,
        "fft_size": fft_size,
        "hop_size": hop_size,
        "in_channels": D,
        "conv_channels": 256,
        "out_channels": 1,
        "ccep_size": 222,
        "kernel_size": 3,
        "use_causal": True,
        "use_tanh": True,
        "use_uvmask": True,
        "use_reference_mag": False,
        "use_weight_norm": True,
        "n_ltv_layers": 2,
        "n_postfilter_layers": 2,
        "n_ltv_postfilter_layers": 4,
        "conv_type": conv_type,
        "postfilter_type": postfilter_type,
        "ltv_postfilter_type": ltv_postfilter_type,
        "scaler_file": None,
    }
    if checkpoint is not None:
        nhv_params["scaler_file"] = str(scalerf)
    return nhv_params


# @pytest.mark.parametrize("checkpoint", [None, checkpoint])
# def test_incremental_nhv_checkpoint(checkpoint):
#     test_incremental_nhv("ddsconv", None, None, checkpoint=checkpoint)


# TODO(k2kobayashi): postfilter_type="conv" does not passed test
@pytest.mark.parametrize("conv_type", ["original", "ddsconv"])
@pytest.mark.parametrize("postfilter_type", [None, "ddsconv"])
@pytest.mark.parametrize("ltv_postfilter_type", [None, "conv", "ddsconv"])
def test_incremental_nhv(
    conv_type, postfilter_type, ltv_postfilter_type, checkpoint=None
):
    nhv_params = get_nhv_params(
        conv_type=conv_type,
        postfilter_type=postfilter_type,
        ltv_postfilter_type=ltv_postfilter_type,
        checkpoint=checkpoint,
    )
    base_net = NeuralHomomorphicVocoder(**nhv_params)
    if checkpoint is not None:
        base_net.load_state_dict(
            torch.load(checkpoint, map_location="cpu")["model"]["generator"],
            strict=True,
        )
    base_net.remove_weight_norm()
    base_net.eval()
    inc_net = IncrementalNeuralHomomorphicVocoder(**nhv_params)
    inc_net.remove_weight_norm()
    inc_net.load_state_dict(base_net.state_dict(), strict=True)
    inc_net.eval()

    # inference by base_net
    # z = torch.arange(B * T * hop_size).float().reshape(B, 1, T * hop_size)
    # x = torch.ones((B, T, D))
    z = torch.randn((B, 1, T * hop_size))
    x = torch.randn((B, T, D))
    f0 = torch.arange(B * T * 1).float().reshape(B, T, 1) * 10
    uv = torch.ones(B, T, 1)
    st = time.time()
    base_out = base_net._forward(z, x, f0, uv)
    assert base_out.size(-1) == T * hop_size
    print("base done")

    # inference by inc_net
    ltv_caches = inc_net.reset_ltv_caches()
    conv_caches = inc_net.reset_caches(z, x, f0, uv, hop_size=hop_size, batch_size=B)

    st = time.time()
    inc_out = []
    for t in range(T):
        y, ltv_caches, conv_caches = inc_net.forward(
            z[..., t * hop_size : (t + 1) * hop_size],
            x[:, t : t + 1],
            f0[:, t : t + 1],
            uv[:, t : t + 1],
            ltv_caches,
            conv_caches,
        )
        inc_out += [y]
    print(f"Inference time / frame using torch: {(time.time() - st) / T}")
    inc_out = torch.cat(inc_out, axis=-1)
    base_out = base_out[..., 2 * hop_size :]
    inc_out = inc_out[..., 2 * hop_size :]
    print("inc done")

    # test for base and inc
    print("print TopK abs difference")
    print(torch.topk(torch.abs(inc_out - base_out), hop_size // 2))
    npt.assert_array_almost_equal(
        inc_out.detach().numpy(),
        base_out.detach().numpy(),
        decimal=3,
    )


@pytest.mark.skipif(not onnxf.exists(), reason="onnx model file does not exist.")
def test_inc_net_onnx():
    nhv_params = get_nhv_params(
        checkpoint=checkpoint,
    )
    base_net = NeuralHomomorphicVocoder(**nhv_params)
    if checkpoint is not None:
        base_net.load_state_dict(
            torch.load(checkpoint, map_location="cpu")["model"]["generator"],
            strict=True,
        )
    base_net.remove_weight_norm()
    base_net.eval()
    inc_net = IncrementalNeuralHomomorphicVocoder(**nhv_params)
    inc_net.remove_weight_norm()
    inc_net.load_state_dict(base_net.state_dict(), strict=True)
    inc_net.eval()

    # z = torch.arange(B * T * hop_size).float().reshape(B, 1, T * hop_size)
    z = torch.randn((B, 1, T * hop_size))
    x = torch.randn((B, T, D))
    f0 = torch.ones(B, T, 1) * 200.0
    uv = torch.ones(B, T, 1)

    # inference by inc_net
    ltv_caches = inc_net.reset_ltv_caches()
    conv_caches = inc_net.reset_caches(z, x, f0, uv, batch_size=B, hop_size=hop_size)
    ltv_caches[0] = torch.rand_like(ltv_caches[0])

    st = time.time()
    inc_out = []
    for t in range(T):
        y, ltv_caches, conv_caches = inc_net.forward(
            z[..., t * hop_size : (t + 1) * hop_size],
            x[:, t : t + 1],
            f0[:, t : t + 1],
            uv[:, t : t + 1],
            ltv_caches,
            conv_caches,
        )
        inc_out += [y]
    print(f"Inference time / frame using torch: {(time.time() - st) / T}")
    inc_out = torch.cat(inc_out, axis=-1)

    input_names = ["z", "x", "f0", "uv"]
    input_names += [f"ltv_cache_{i}" for i in range(len(ltv_caches))]
    input_names += [f"conv_cache_{i}" for i in range(len(conv_caches))]
    output_names = ["y"]
    output_names += [f"new_ltv_cache_{i}" for i in range(len(ltv_caches))]
    output_names += [f"new_conv_cache_{i}" for i in range(len(conv_caches))]
    z_t = z[..., 0 : 1 * hop_size]
    x_t = x[:, 0 : 0 + 1]
    f0_t = f0[:, 0 : 0 + 1]
    uv_t = uv[:, 0 : 0 + 1]

    torch.onnx.export(
        inc_net,
        (
            z_t,
            x_t,
            f0_t,
            uv_t,
            ltv_caches,
            conv_caches,
        ),
        onnxf,
        verbose=False,
        input_names=input_names,
        output_names=output_names,
        opset_version=13,
    )

    sess = ort.InferenceSession(str(onnxf))
    inputs, cache_idx = {}, {}
    for i in sess.get_inputs():
        inputs.update({i.name: np.zeros(i.shape).astype(np.float32)})
        if "cache" in i.name:
            cache_idx[i.name] = None
    for i, o in enumerate(sess.get_outputs()):
        for k in cache_idx.keys():
            if f"new_{k}" == o.name:
                cache_idx[k] = i
    inputs.update(
        {
            "z": z_t.cpu().numpy(),
            "x": x_t.cpu().numpy(),
            "f0": f0_t.cpu().numpy(),
            "uv": uv_t.cpu().numpy(),
        }
    )

    st = time.time()
    onnx_outs = []
    for t in range(T):
        inputs.update(
            {
                "z": z[..., t * hop_size : (t + 1) * hop_size].cpu().numpy(),
                "x": x[:, t : t + 1].cpu().numpy(),
                "f0": f0[:, t : t + 1].cpu().numpy(),
                "uv": uv[:, t : t + 1].cpu().numpy(),
            }
        )
        outputs = sess.run(None, inputs)  # noqa
        for k, v in cache_idx.items():
            inputs[k] = outputs[v]
        onnx_outs += [outputs[0]]
    print(f"Elapsed time: {(time.time() - st) / T}")

    onnx_out = np.hstack(onnx_outs).flatten()
    npt.assert_array_almost_equal(
        inc_out[..., hop_size:].squeeze().detach().numpy(),
        onnx_out[..., hop_size:],
        decimal=3,
    )


@pytest.mark.skipif(not onnxf.exists(), reason="onnx model file does not exist.")
def test_decoder_onnx():
    # load model
    sess = ort.InferenceSession(str(onnxf))
    inputs, cache_idx = {}, {}
    for i in sess.get_inputs():
        inputs.update({i.name: np.zeros(i.shape).astype(np.float32)})
        if "cache" in i.name:
            cache_idx[i.name] = None
    for i, o in enumerate(sess.get_outputs()):
        for k in cache_idx.keys():
            if f"new_{k}" == o.name:
                cache_idx[k] = i

    # load feature

    with h5py.File(h5f, "r") as f:
        cf0 = f["cf0"][:][np.newaxis]
        x = f["mlfb"][:][np.newaxis]
        uv = f["uv"][:][np.newaxis]
    z = torch.normal(0, 0.03, (1, 1, cf0.shape[1] * hop_size)).cpu().numpy()

    inputs.update(
        {
            "z": z[..., :1, :hop_size],
            "x": x[..., :1, :],
            "f0": cf0[..., :1, :1],
            "uv": uv[..., :1, :1],
        }
    )
    st = time.time()
    onnx_outs = []
    for t in range(cf0.shape[1]):
        inputs.update(
            {
                "z": z[..., t * hop_size : (t + 1) * hop_size],
                "x": x[..., t : t + 1, :],
                "f0": cf0[..., t : t + 1, :1],
                "uv": uv[..., t : t + 1, :1],
            }
        )
        outputs = sess.run(None, inputs)  # noqa
        for k, v in cache_idx.items():
            inputs[k] = outputs[v]
        onnx_outs += [outputs[0]]
    print(f"Elapsed time: {(time.time() - st) / cf0.shape[1]}")

    wav = np.vstack([onnx_outs]).flatten()
    sf.write("test.wav", wav, 24000)
