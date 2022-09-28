from __future__ import print_function
import numpy as np
import cupy as cp
from deblur.DeblurGANv2.predict import Predictor as DeblurGAN


class DeblurNet:
    def __init__(self, network_type='DeblurGAN', 
                 weights_path='deblur/DeblurGANv2/pretrained/best_fpn.h5', device='cuda'):
        self.device = device
        self.type = network_type
        self.weights = weights_path

    def __call__(self, img):  #img is 3-channel RGB , [0,1]

        if self.type == 'DeblurGAN':
            DB = DeblurGAN(weights_path=self.weights)
        # elif self.type == 'DMPHN':
        #     DN = DMPHN()

        out = DB(img)

        return np.float32(np.clip(out, 0, 1))


