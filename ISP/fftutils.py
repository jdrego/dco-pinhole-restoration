import numpy as np
import utils

def FFT(src):
    fft_src = np.zeros_like(src, dtype=complex)
    for c in range(src.shape[2]):
        fft_src[:, :, c] = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(src[:,:,c])))
    return fft_src

def plot_fft(src, main_title=None, save=None):
    src_fft_abs = np.abs(FFT(src))
    utils.print_info(src_fft_abs)
    utils.imshow([20*np.log(src_fft_abs[:,:,0]), 20*np.log(src_fft_abs[:,:,1]), 20*np.log(src_fft_abs[:,:,2])],
                 ['FFT_R', 'FFT_G', 'FFT_B'], dpi=200, axis=True, main_title=main_title, save=save)

def plot_fft_cross(src, crop=None):
    src_fft_abs = np.abs(FFT(src))
    h, w = src_fft_abs.shape[:2]
    if crop != None:
        src_fft_abs = src_fft_abs[h//2-crop:h//2+crop, w//2-crop:w//2+crop, :]
    h, w = src_fft_abs.shape[:2]
    titles = ['Red', 'Green', 'Blue']
    for c in range(3):
        utils.plot([(np.arange(-w//2, w//2), ((src_fft_abs))[h//2, :, c]),
                    (np.arange(-h//2, h//2), ((src_fft_abs))[:, w//2, c])], 
                    title=[titles[c] + 'Horizontal', titles[c] + 'Vertical'])