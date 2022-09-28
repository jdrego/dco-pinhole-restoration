import os, sys
import glob
import rawpy
import cv2
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import ISP as isp
import ISP.utils as utils
from denoise import LPfilter, DenoiseNet
from deblur import DeblurNet

def main(filenames, out_path, display, device):
    for fpath in filenames:
        fn = os.path.splitext(os.path.basename(fpath))[0]
        print(fn)

        raw = rawpy.imread(fpath)
        pattern = raw.raw_pattern
        byr = raw.raw_image_visible.copy()

        ############# Preprocessing RAW image #############
        blc = isp.black_level_correction(byr, black_level=raw.black_level_per_channel,
                                         white_level=raw.white_level, pattern=pattern,
                                         clip_range=[0, raw.white_level])

        rgb = isp.demosaic_cv2(blc, pattern).astype('float32') / raw.white_level

        # Display input image (postprocess for visualization)
        if display:
            utils.imshow(isp.postprocess(rgb, raw), titles='Input Pinhole Image')

        ################## Denoise ##################
        print('Denoising...')

        LPFilter = LPfilter(device='cuda')
        denoiseNet = DenoiseNet(device='cuda')

        rgb_denoise = LPFilter(rgb)
        rgb_denoise = denoiseNet(rgb_denoise)
        
        if display:
            utils.imshow(isp.postprocess(rgb_denoise, raw), titles='Denoised Pinhole Image')

        ################## Deblur ##################
        print('Debluring...')

        deblurNet = DeblurNet(device='cuda')

        rgb_deblur = deblurNet(rgb_denoise*255, None)
        
        if display:
            utils.imshow(isp.postprocess(rgb_deblur, raw), titles='Deblured Pinhole Image')

        ################## Save Output Image ##################
        cv2.imwrite(os.path.join(out_path, fn + '_deblur.jpg'), 
                    np.uint8(np.clip(isp.postprocess(rgb_deblur), 0, 1) * 255)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_path', type=str, default='./data/real_scenes/',
                        help='Path to file or directory of files')
    parser.add_argument('--out_path', type=str, default=None,
                        help='Path to save output image')
    parser.add_argument('--display', type=bool, default=True,
                        help='Display images at each step. Default: True')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Specify whether to use CPU "cpu" or GPU "cuda"')
    args = parser.parse_args()

    if os.path.isfile(args.in_path):
        filenames = [args.in_path]
    elif os.path.isdir(args.in_path):
        filenames = sorted(glob.glob(args.in_path + '*.RW2'))
        
    if args.out_path == None:
            args.out_path = './output/'
    utils.makedirs(args.out_path)

    main(filenames, args.out_path, args.display, args.device)
