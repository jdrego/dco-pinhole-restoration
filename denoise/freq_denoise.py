import os, sys, glob
import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
import imageio as io
import argparse
import cv2
import time
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
import utils


def draw_circle(shape, diameter):
    '''
    Input:
    shape    : tuple (height, width)
    diameter : scalar
    
    Output:
    np.array of shape  that says True within a circle with diameter =  around center 
    '''
    assert len(shape) == 2
    TF = np.zeros(shape,dtype=np.bool)
    center = np.array(TF.shape)/2.0

    for iy in range(shape[0]):
        for ix in range(shape[1]):
            TF[iy,ix] = (iy- center[0])**2 + (ix - center[1])**2 < diameter **2
    return TF
    
def apply_filter(fft_img_channel, filter_circle):
    xp = cp.get_array_module(fft_img_channel)
    temp = xp.zeros(fft_img_channel.shape[:2], dtype=complex)
    temp[filter_circle] = fft_img_channel[filter_circle]
    return temp

def inv_FFT_all_channel(fft_img):
    xp = cp.get_array_module(fft_img)
    img_reco = []
    for c in range(fft_img.shape[2]):
        img_reco.append(xp.fft.ifft2(xp.fft.ifftshift(fft_img[:,:,c])))
    img_reco = xp.array(img_reco)
    img_reco = xp.transpose(img_reco,(1,2,0))
    return img_reco

def process(img, LP_mask=None, freq_diam=200):
    xp = cp.get_array_module(img)
    shape = img.shape[:2]
    # print(shape, type(freq_diam))
    # Create low-pass Mask
    if LP_mask is None:
        LP_mask = xp.array(draw_circle(shape=img.shape[:2], diameter=freq_diam[0]))
    if type(freq_diam) != list and type(freq_diam) != tuple:
        freq_diam = [freq_diam] * img.shape[2]
    # print(freq_diam)
    
    # FFT input image for each channel
    fft_img = xp.zeros_like(img, dtype=complex)
    for c in range(fft_img.shape[2]):
        fft_img[:,:,c] = xp.fft.fftshift(np.fft.fft2(img[:, :, c]))
    
    fft_img_filtered = []
    ## for each channel, pass filter
    for c in range(fft_img.shape[2]):
        print(freq_diam[c])
        fft_img_channel = fft_img[..., c]
        # LP_mask = xp.array(draw_circle(shape=img.shape[:2], diameter=freq_diam[c]))
        temp = apply_filter(fft_img_channel, LP_mask)
        fft_img_filtered.append(temp)
        
    fft_img_filtered = xp.array(fft_img_filtered).transpose(1,2,0)

    # abs_fft_img           = xp.abs(fft_img)
    # abs_fft_img_filtered  = xp.abs(fft_img_filtered)
    
    # img_reco           = inv_FFT_all_channel(fft_img)
    img_reco_filtered  = inv_FFT_all_channel(fft_img_filtered)
    
    return xp.abs(img_reco_filtered)

def output_process(img, crop_dim):
    img = (img/img.max())**(1/2.2) * 255.0
    if crop_dim != None:
        h, w, c = img.shape
        h1, w1 = args.crop_dim
        img = img[h//2 - h1:h//2 + h1, w//2 - w1:w//2 + w1, [2,1,0]]

    return np.uint8(np.clip(img, 0, 255))

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', type=str, default='../data/natural_scenes/processed/P1026175.npy',
                        help='Path to file or directory of files')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='Path to save output image')
    parser.add_argument('--freq_diameter', type=int, default=200,
                        help='Specify diameter of frequency mask to use for low-pass filter')
    parser.add_argument("--crop_dim", nargs='*', type=int, default=None)
    parser.add_argument('--device', type=str, default='cuda',
                        help='Specify whether to use CPU "cpu" or GPU "cuda"')
    args = parser.parse_args()
    
    if os.path.isfile(args.in_dir):
        filenames = [args.in_dir]
        if args.out_dir == None:
            args.out_dir = os.path.dirname(args.in_dir) + '/denoised_LP/'
    elif os.path.isdir(args.in_dir):
        filenames = sorted(glob.glob(args.in_dir + '*.npy'))
        if args.out_dir == None:
            args.out_dir = args.in_dir + '/denoised_LP/'
    else:
        raise ValueError('Input file/path does not exist: {}'.format(args.in_dir))
    utils.makedirs([args.out_dir, 
                    args.out_dir + 'npy/', args.out_dir + 'jpg'])
    
    print('-'*50)
    print('Settings:')
    print('\tin_dir: \t', args.in_dir)
    print('\tout_dir: \t', args.out_dir)
    print('\tfreq_diameter: \t', args.freq_diameter)
    print('\tcrop_dim: \t', args.crop_dim)
    print('\tdevice: \t', args.device)
    print()
    print('Number of files:', len(filenames))
    print('-'*50)
    
    for in_dir in tqdm(filenames):
        fn = os.path.splitext(os.path.basename(in_dir))[0]
        # PSNR: 22.31, SSIM: 0.634
        
        # print(args.out_dir)
        ## Load image
        img = np.load(in_dir)
        img = cv2.resize(img, (1920,1080))
        # img = cv2.imread(in_dir).astype('float32')[:,:,::-1] / 255.0
        if img.dtype == np.uint16:
            img = img.astype('float32') / (2**14 - 1)
        # Print and visualize input image    
        utils.print_info(img, fn)
        # cv2.imwrite(args.out_dir + 'input/' + fn + '.jpg', 
        #            output_process(img, args.crop_dim))
        # utils.imshow((img)**(1/2.2), titles='Blur', dpi=200)
        # img /= img.max()
        # img = img**(1/2.2)
        # img = img * (0.2/np.mean(img))
        if in_dir == filenames[0]:
            start_time = time.time()
            LP_mask = draw_circle(shape=img.shape[:2], diameter=args.freq_diameter)
            end_time = time.time()
            print(end_time - start_time, 'seconds')
        # Move to GPU if desired
        if args.device == 'cuda':
            img = cp.array(img)
            LP_mask = cp.array(LP_mask)
        
        start_time = time.time()
        ## Low-pass Filter
        img_denoise = process(img, LP_mask, freq_diam=args.freq_diameter)
        end_time = time.time()
        print(end_time - start_time)
        # Back to CPU
        if args.device == 'cuda':
            img_denoise = cp.asnumpy(img_denoise)
            img = cp.asnumpy(img)
            
        # img_denoise = img_denoise**(1/2.2)
        # img_denoise = img_denoise * (0.2/np.mean(img_denoise))
        # img_denoise = np.clip(img_denoise, 0, 1)
        # img_denoise /= img_denoise.max()
        # img_denoise = img_denoise**(1/2.2)
        
        if args.crop_dim != None:
            h, w, c = img_denoise.shape
            h1, w1 = args.crop_dim
            img_denoise = img_denoise[h//2 - h1:h//2 + h1, w//2 - w1:w//2 + w1, :]
            img_denoise = np.clip(img_denoise, 0, 1)
        ## Save Denoised Image
        np.save(args.out_dir + 'npy/{}_denoised_{}.npy'.format(fn, args.freq_diameter), 
                img_denoise.astype('float32'))
        cv2.imwrite(args.out_dir + 'jpg/{}_denoised_{}.jpg'.format(fn, args.freq_diameter), 
                    np.uint8((img_denoise) * 255.0)[:,:,::-1])
        
        # img = img**(1/2.2)
        # img = img * (0.2/np.mean(img))
        img = np.clip(img, 0, 1)
        cv2.imwrite(args.out_dir + 'jpg/{}.jpg'.format(fn), 
                    np.uint8((img) * 255.0)[:,:,::-1])