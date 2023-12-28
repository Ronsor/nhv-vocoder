[![CI](https://github.com/k2kobayashi/neural-homomorphic-vocoder/actions/workflows/ci.yaml/badge.svg)](https://github.com/k2kobayashi/neural-homomorphic-vocoder/actions/workflows/ci.yaml)
[![PyPI version](https://badge.fury.io/py/neural-homomorphic-vocoder.svg)](https://badge.fury.io/py/neural-homomorphic-vocoder)
[![Downloads](https://pepy.tech/badge/neural-homomorphic-vocoder)](https://pepy.tech/project/neural-homomorphic-vocoder)

# neural-homomorphic-vocoder

A neural vocoder based on source-filter model called neural homomorphic vocoder

# Install

```shell
pip install neural-homomorphic-vocoder
```

# Usage

Usage for NeuralHomomorphicVocoder class
- Input
    - z: Gaussian noise
    - x: mel-filterbank
    - cf0: continuous f0
    - uv: u/v symbol

```python
import torch
from nhv import NeuralHomomorphicVocoder

net = NeuralHomomorphicVocoder(
        fs=24000,             # sampling frequency
        fft_size=1024,        # size for impuluse responce of LTV
        hop_size=256,         # hop size in each mel-filterbank frame
        in_channels=80,       # input channels (i.e., dimension of mel-filterbank)
        conv_channels=256,    # channel size of LTV filter
        ccep_size=222,        # output ccep size of LTV filter      
        out_channels=1,       # output size of network
        kernel_size=3,        # kernel size of LTV filter
        dilation_size=1,      # dilation size of LTV filter
        group_size=8,         # group size of LTV filter
        fmin=80,              # min freq. for melspc 
        fmax=7600,            # max freq. for melspc (recommend to use full-band)
        roll_size=24,         # frame size to get median to estimate logspc from melspc
        n_ltv_layers=3,       # # layers for LTV ccep generator
        n_postfilter_layers=4,     # # layers for output postfilter 
        n_ltv_postfilter_layers=1, # # layers for LTV postfilter (if ddsconv)
        harmonic_amp=0.1,     # amplitude of sinusoidals
        noise_std=0.03        # standard deviation of Gaussian noise
        use_causal=False,     # use causal conv LTV filter
        use_reference_mag=False,   # use reference logspc calculated from melspc
        use_tanh=False,       # apply tanh to output else linear
        use_uvmask=False,     # apply uv-based mask to harmonic
        use_weight_norm=True, # apply weight norm to conv1d layer
        conv_type="original"  # LTV generator network type ["original", "ddsconv"]
        postfilter_type=None, # postfilter network type ["None", "normal", "ddsconv"]
        ltv_postfilter_type=None,  # LTV postfilter network type \
                                   # ["None", "normal", "ddsconv"]
        ltv_postfilter_kernel_size=128  # kernel_size for LTV postfilter
        scaler_file=None      # path to .pkl for internal scaling of melspc
                              # (dict["mlfb"] = sklearn.preprocessing.StandardScaler)

    conv_type = "original"
    postfilter_type = "ddsconv"
    ltv_postfilter_type = "conv"
    ltv_postfilter_kernel_size = 128
    scaler_file = None


)

B, T, D = 3, 100, in_channels   # batch_size, n_frames, n_mels
z = torch.randn(B, 1, T * hop_size)
x = torch.randn(B, T, D)
cf0 = torch.randn(B, T, 1)
uv = torch.randn(B, T, 1)
y = net(z, torch.cat([x, cf0, uv], dim=-1))  # z: (B, 1, T * hop_size), c: (B, D+2, T)
y = net._forward(z, x, cf0, uv)
y = net.inference(c)  # for evaluation
```

# Features

- Train using [kan-bayashi/ParallelWaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN) with continuous F0 and uv symbols
- Support depth-wise separable convolution
- Support incremental inference

# References

```bibtex
@article{liu20,
  title={Neural Homomorphic Vocoder},
  author={Z.~Liu and K.~Chen and K.~Yu},
  journal={Proc. Interspeech 2020},
  pages={240--244},
  year={2020}
}
```
