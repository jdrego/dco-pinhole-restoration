from __future__ import print_function
import numpy as np
import cupy as cp
from denoise import freq_denoise
from denoise.ffdnet.models import FFDNet


class LPfilter:
    def __init__(self, freq_diameter=200, device='cuda'):
        self.device = device
        self.freq_d = freq_d
        
    def __call__(self, img):
        if self.device == 'cuda':
            img = cp.asarray(img)
        img_filter = freq_denoise.process(img, freq_diam=self.freq_d)
        out = cp.asnumpy(img_filter)

        del img
        del img_filter
        cp._default_memory_pool.free_all_blocks()

        return np.float32(np.clip(out, 0, 1))


class DenoiseNet:
    def __init__(self, network_type='FFDNet', device='cuda'):
        self.device = device
        self.type = network_type

    def __call__(self, img):  #img is 3-channel RGB , [0,1]
        if self.type=='FFDNet':
            DN = FFDNet(num__input_channels=img.shape[2])
        # elif self.type == 'FCAIDE':
        #     DN = FCAIDE()

        out = DN(img)

        return np.float32(np.clip(out, 0, 1))


