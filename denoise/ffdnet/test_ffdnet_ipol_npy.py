"""
Denoise an image with the FFDNet denoising method

Copyright (C) 2018, Matias Tassano <matias.tassano@parisdescartes.fr>

This program is free software: you can use, modify and/or
redistribute it under the terms of the GNU General Public
License as published by the Free Software Foundation, either
version 3 of the License, or (at your option) any later
version. You should have received a copy of this license along
this program. If not, see <http://www.gnu.org/licenses/>.
"""
import os
import glob
import argparse
import time
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm
from models import FFDNet
from utils import batch_psnr, normalize, init_logger_ipol, \
                variable_to_cv2_image, remove_dataparallel_wrapper, is_rgb
import time

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def center_crop(img, crop):
    if type(crop) != list or type(crop) != tuple:
        crop = [crop, crop]
    h, w = img.shape[:2]
    return img[h//2-crop[0]:h//2+crop[0], w//2-crop[1]:w//2+crop[1]]

def test_crops(imorig, model, crop_factor, noise_sigma, dtype):
    h, w = imorig.shape[2:]
    img_out = np.zeros(imorig.shape).astype('float32')
    m, n = [h // crop_factor,
             w // crop_factor]
    
    for i in range(crop_factor):
        for j in range(crop_factor):
            with torch.no_grad(): # PyTorch v0.4.0
                patchorig = torch.Tensor(imorig[:, :, i*m:i*m + m, j*n:j*n+n])
                imnoisy = patchorig.clone()
                patchorig, imnoisy = Variable(patchorig.type(dtype)), \
                                  Variable(imnoisy.type(dtype))
                nsigma = Variable(torch.FloatTensor([noise_sigma]).type(dtype))

                # Measure runtime
                start_t = time.time()

                # Estimate noise and subtract it to the input image
                im_noise_estim = model(imnoisy, nsigma)
                outim = torch.clamp(imnoisy-im_noise_estim, 0.)#, 1.)
                img_out[:, :, i*m:i*m + m, j*n:j*n+n] = (outim.data.cpu().numpy())
                stop_t = time.time()

    return img_out[0].transpose(1,2,0)

def test_ffdnet(**args):
    r"""Denoises an input image with FFDNet
    """
    # Init logger
    logger = init_logger_ipol()

    # Check if input exists and if it is RGB
    try:
        rgb_den = True #is_rgb(args['input'])
    except:
        raise Exception('Could not open the input image')

    # Open image as a CxHxW torch.Tensor
    if rgb_den:
        in_ch = 3
        model_fn = 'models/net_rgb.pth'
        imorig = np.load(args['input']).astype('float32')
        h, w = imorig.shape[:2]
        
        if args['crop_dim'] != None:
            h, w, c = imorig.shape
            h1, w1 = args['crop_dim']
            imorig = imorig[h//2 - h1:h//2 + h1, w//2 - w1:w//2 + w1, :]
            imorig = np.clip(imorig, 0, 1)
        # from HxWxC to CxHxW, RGB image
        imorig = imorig.transpose(2, 0, 1)
    else:
        # from HxWxC to  CxHxW grayscale image (C=1)
        in_ch = 1
        model_fn = 'models/net_gray.pth'
        imorig = cv2.imread(args['input'], cv2.IMREAD_GRAYSCALE)
        imorig = np.expand_dims(imorig, 0)
    imorig = np.expand_dims(imorig, 0)

    # Handle odd sizes
    expanded_h = False
    expanded_w = False
    sh_im = imorig.shape
    if sh_im[2]%2 == 1:
        expanded_h = True
        imorig = np.concatenate((imorig, \
                imorig[:, :, -1, :][:, :, np.newaxis, :]), axis=2)

    if sh_im[3]%2 == 1:
        expanded_w = True
        imorig = np.concatenate((imorig, \
                imorig[:, :, :, -1][:, :, :, np.newaxis]), axis=3)

    

    # Absolute path to model file
    model_fn = os.path.join(os.path.abspath(os.path.dirname(__file__)), \
                model_fn)

    # Create model
    # print('Loading model ...\n')
    net = FFDNet(num_input_channels=in_ch)

    # Load saved weights
    if args['cuda']:
        state_dict = torch.load(model_fn)
        device_ids = [0]
        model = nn.DataParallel(net, device_ids=device_ids).cuda()
    else:
        state_dict = torch.load(model_fn, map_location='cpu')
        # CPU mode: remove the DataParallel wrapper
        state_dict = remove_dataparallel_wrapper(state_dict)
        model = net
    model.load_state_dict(state_dict)

    # Sets the model in evaluation mode (e.g. it removes BN)
    model.eval()

    # Sets data type according to CPU or GPU modes
    if args['cuda']:
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor
  

    start = time.time()
    outimg = test_crops(imorig, model, 1, args['noise_sigma'], dtype)
    print(time.time() - start)
    noisyimg = imorig[0].copy().transpose(1,2,0)
    # Save images
    out_path = args['output']
    os.makedirs(out_path + 'npy/', exist_ok=True)
    os.makedirs(out_path + 'jpg/', exist_ok=True)
    fn = os.path.splitext(os.path.basename(args['input']))[0]
    if not args['dont_save_results']:
        np.save(out_path + 'npy/{}_ffd{}.npy'.format(fn, int(args['noise_sigma']*255)),
                (outimg))
        # noisyimg = variable_to_cv2_image(imnoisy)
        # outimg = variable_to_cv2_image(outim)
        noisyimg = np.uint8(np.clip(noisyimg, 0, 1) * 255)[:,:,::-1]
        outimg = np.uint8(np.clip(outimg, 0, 1) * 255)[:,:,::-1]
        
        cv2.imwrite(out_path + "jpg/{}_noisy.jpg".format(fn), noisyimg)
        cv2.imwrite(out_path + "jpg/{}_ffd{}.jpg".format(fn, int(args['noise_sigma']*255)), outimg)
        if args['add_noise']:
            cv2.imwrite(out_path + "jpg/noisy_diff.jpg", variable_to_cv2_image(diffnoise))
            cv2.imwrite(out_path + "jpg/ffdnet_diff.jpg", variable_to_cv2_image(diffout))

if __name__ == "__main__":
    import time
    # Parse arguments
    parser = argparse.ArgumentParser(description="FFDNet_Test")
    parser.add_argument('--add_noise', type=str, default="False")
    parser.add_argument("--input", type=str, default="", \
                        help='path to input image')
    parser.add_argument('--output', type=str, default='./output/',
                        help='path to save output')
    parser.add_argument("--suffix", type=str, default="", \
                        help='suffix to add to output name')
    parser.add_argument("--noise_sigma", type=float, default=25, \
                        help='noise level used on test set')
    parser.add_argument("--dont_save_results", action='store_true', \
                        help="don't save output images")
    parser.add_argument("--no_gpu", action='store_true', \
                        help="run model on CPU")
    parser.add_argument("--crop_dim", nargs='*', type=int, default=None)
    argspar = parser.parse_args()
    # Normalize noises ot [0, 1]
    argspar.noise_sigma /= 255.

    # String to bool
    argspar.add_noise = (argspar.add_noise.lower() == 'true')

    # use CUDA?
    argspar.cuda = not argspar.no_gpu and torch.cuda.is_available()

    print("\n### Testing FFDNet model ###")
    print("> Parameters:")
    for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
        print('\t{}: {}'.format(p, v))
    print('\n')

    in_path = argspar.input
    if os.path.isfile(in_path) == True:
        start = time.time()
        test_ffdnet(**vars(argspar))
        end = time.time()
        print(end-start)

    elif os.path.isdir(in_path) == True:
        filepaths = sorted(glob.glob(in_path + '*.npy'))
        for f in tqdm(filepaths[:]):
            argspar.input = f
            test_ffdnet(**vars(argspar))
   

    else:
        raise ValueError('Input is not a file or directory: {}'.format(in_path))
  

