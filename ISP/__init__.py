#%%
import numpy as np
import cupy as cp
import math
import cv2
from tqdm import tqdm

def getBayerPattern(array):
    string = []
    array = array.reshape(array.size)
    for i in range(len(array)):
        if array[i] == 0: 
            string.append('r')
        elif array[i] == 1 or array[i] == 3:
            string.append('g')
        elif array[i] == 2:
            string.append('b')
        else:
            raise ValueError('Invalid value in bayer pattern array. \
                Values must be between 0 and 3 corresponding to R, Gr, B, Gb')
    return ''.join(string)

def getBitDepth():
    pass

def rgbg_reorder(array, pattern):
    pattern = pattern.reshape(-1)
    return np.array(array)[pattern]

def bayer2rgbg(bayer, pattern, num_channels=4):
    """ Pack single channel Bayer RAW image into RGrBGb 4 channel array

    Args:
        bayer (np.array): Single channel bayer image of shape (H, W)
        pattern (np.array): Raw Bayer 2x2 pattern with numbers corresponding to R Gr B Gb order.
                            e.g. pattern = [[0, 1]  --> [[R Gr] 
                                            [3, 2]]      [Gb B]]
                                            
                                 pattern = [[2, 3]  --> [[B Gb] 
                                            [1, 0]]      [Gr R]]
    Returns:
        np.array: R Gr B Gb 4-channel image of shape (H//2, W//2, 4)
    """
    assert len(bayer.shape) == 2
    im = np.expand_dims(bayer, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    # First pixel coordinates RGrBGb colors
    R = np.argwhere(pattern == 0)[0]
    Gr = np.argwhere(pattern == 1)[0]
    B = np.argwhere(pattern == 2)[0]
    Gb = np.argwhere(pattern == 3)[0]

    if num_channels == 4:
        out = np.concatenate((im[R[0]::2,   R[1]::2, :],
                              im[Gr[0]::2,  Gr[1]::2, :],
                              im[B[0]::2,   B[1]::2, :],
                              im[Gb[0]::2,  Gb[1]::2, :]), axis=2)
    elif num_channels == 3:
        out = np.concatenate((im[R[0]::2,   R[1]::2, :],
                              (im[Gr[0]::2, Gr[1]::2, :] + im[Gb[0]::2, Gb[1]::2, :]) / 2,
                              im[B[0]::2,   B[1]::2, :]), axis=2)
    return out.astype(bayer.dtype)

def rgbg2bayer(rgbg, pattern, num_channels):
    """ Convert 4-channel R Gr B Gb image to single channel Bayer image

    Args:
        rgbg (np.array): 4-channel R Gr B Gb image of shape (H, W)
        pattern (np.array): Raw Bayer 2x2 pattern with numbers corresponding to R Gr B Gb order.
                            e.g. pattern = [[0, 1]  --> [[R Gr] 
                                            [3, 2]]      [Gb B]]
                                            
                                 pattern = [[2, 3]  --> [[B Gb] 
                                            [1, 0]]      [Gr R]]
    Returns:
        np.array: Single-channel image of shape (H*2, W*2)
    """
    assert (len(rgbg.shape) == 3) & (rgbg.shape[2] >= 3) # Must be 3 or 4-channel array
    h, w = rgbg.shape[:2]
    bayer = np.zeros((h * 2, w * 2)) # Initialize array to store bayer image
    
    # First pixel coordinates RGrBGb colors
    R = np.argwhere(pattern == 0)[0]
    Gr = np.argwhere(pattern == 1)[0]
    B = np.argwhere(pattern == 2)[0]
    Gb = np.argwhere(pattern == 3)[0]
    if num_channels == 4:
        bayer[R[0]::2,  R[1]::2] = rgbg[..., 0]
        bayer[Gr[0]::2, Gr[1]::2] = rgbg[..., 1]
        bayer[B[0]::2,  B[1]::2] = rgbg[..., 2]
        bayer[Gb[0]::2, Gb[1]::2] = rgbg[..., 3]
    elif num_channels == 3:
        bayer[R[0]::2,  R[1]::2] = rgbg[..., 0]
        bayer[Gr[0]::2, Gr[1]::2] = rgbg[..., 1]
        bayer[B[0]::2,  B[1]::2] = rgbg[..., 2]
        bayer[Gb[0]::2, Gb[1]::2] = rgbg[..., 1]
    
    return bayer.astype(rgbg.dtype)


def black_level_correction(raw, black_level, white_level, pattern, clip_range=None):
    globals()['np'] = cp.get_array_module(raw)
    if type(white_level) is not (list or tuple) :
        white_level = [white_level, white_level, white_level, white_level]
    
    # Reorder levels from RGBG to camera's pattern
    black_level = rgbg_reorder(black_level, pattern)
    white_level = rgbg_reorder(white_level, pattern)
    raw = raw.astype('float32')
    blc = np.zeros(raw.shape).astype('float32')
    
    blc[::2, ::2]   = (raw[::2, ::2] - black_level[0]) / (white_level[0] - black_level[0])
    blc[::2, 1::2]  = (raw[::2, 1::2] - black_level[1]) / (white_level[1] - black_level[1])
    blc[1::2, ::2]  = (raw[1::2, ::2] - black_level[2]) / (white_level[2] - black_level[2])
    blc[1::2, 1::2] = (raw[1::2, 1::2] - black_level[3]) / (white_level[3] - black_level[3])
    blc *= white_level[0]
    
    if clip_range is not None:
        blc = np.clip(blc, clip_range[0], clip_range[1])
    return blc.astype('uint16')

# =============================================================
# function: bad_pixel_correction
#   correct for the bad (dead, stuck, or hot) pixels
# =============================================================
def bad_pixel_correction(data, neighborhood_size):
    globals()['np'] = cp.get_array_module(data)
    print("----------------------------------------------------")
    print("Running bad pixel correction...")

    if ((neighborhood_size % 2) == 0):
        raise ValueError("neighborhood_size shoud be odd number, recommended value 3")

    # convert to float32 in case they were not
    # Being consistent in data format to be float32
    data = data.astype('float32')

    # Separate out the quarter resolution images
    D = {} # Empty dictionary
    D[0] = data[::2, ::2]
    D[1] = data[::2, 1::2]
    D[2] = data[1::2, ::2]
    D[3] = data[1::2, 1::2]

    # number of pixels to be padded at the borders
    no_of_pixel_pad = math.floor(neighborhood_size / 2.)

    for idx in range(0, len(D)): # perform same operation for each quarter

        # display progress
        print("bad pixel correction: Quarter " + str(idx+1) + " of 4")

        img = D[idx]
        height, width = img.shape[:2]

        # pad pixels at the borders
        img = np.pad(img, \
                     (no_of_pixel_pad, no_of_pixel_pad),\
                     'reflect') # reflect would not repeat the border value

        for i in tqdm(range(no_of_pixel_pad, height + no_of_pixel_pad)):
            for j in range(no_of_pixel_pad, width + no_of_pixel_pad):

                # save the middle pixel value
                mid_pixel_val = img[i, j]

                # extract the neighborhood
                neighborhood = img[i - no_of_pixel_pad : i + no_of_pixel_pad+1,\
                                   j - no_of_pixel_pad : j + no_of_pixel_pad+1]

                # set the center pixels value same as the left pixel
                # Does not matter replace with right or left pixel
                # is used to replace the center pixels value
                neighborhood[no_of_pixel_pad, no_of_pixel_pad] = neighborhood[no_of_pixel_pad, no_of_pixel_pad-1]

                min_neighborhood = np.min(neighborhood)
                max_neighborhood = np.max(neighborhood)

                if (mid_pixel_val < min_neighborhood):
                    img[i,j] = min_neighborhood
                elif (mid_pixel_val > max_neighborhood):
                    img[i,j] = max_neighborhood
                else:
                    img[i,j] = mid_pixel_val
        #pbar1.update()
        # Put the corrected image to the dictionary
        D[idx] = img[no_of_pixel_pad : height + no_of_pixel_pad,\
                     no_of_pixel_pad : width + no_of_pixel_pad]

    # Regrouping the data
    data[::2, ::2]   = D[0]
    data[::2, 1::2]  = D[1]
    data[1::2, ::2]  = D[2]
    data[1::2, 1::2] = D[3]

    return data.astype('uint16')

def whitebalance_bayer(raw, wb, pattern):
    """TODO:[summary]

    Args:
        raw ([type]): [description]
        wb ([type]): [description]
        pattern ([type]): [description]

    Raises:
        AssertionError: [description]

    Returns:
        [type]: [description]
    """
    if len(raw.shape) != 2:
        raise AssertionError('src must be single-channel bayer image of shape (H,W). \
            If src is a 3-channel RGB image, use whitebalance_rgb() instead')
    
    wb = np.array(wb)
    if len(wb) == 4 and wb[3] == 0:
        wb[3] = wb[1]
    if wb[1] == 256:
        wb /= wb[1]
         
    wb = rgbg_reorder(wb, pattern)
    raw = raw.astype('float32')
    
    raw[::2, ::2]   = raw[::2, ::2] * wb[0]
    raw[::2, 1::2]  = raw[::2, 1::2] * wb[1]
    raw[1::2, ::2]  = raw[1::2, ::2] * wb[2]
    raw[1::2, 1::2] = raw[1::2, 1::2] * wb[3]

    raw = np.clip(raw, 0, None)
    return raw.astype('uint16')


def whitebalance_rgb(src, wb):
    """TODO:[summary]

    Args:
        src ([type]): [description]
        wb ([type]): [description]

    Raises:
        AssertionError: [description]

    Returns:
        [type]: [description]
    """
    if src.shape[2] != 3 and src.shape[2] != 4:
        raise AssertionError('src must be 3-channel RGB images. \
            If src is a 2d bayer image, use whitebalance_bayer() instead')
    
    wb = np.array(wb)
    if len(wb) == 4 and wb[3] == 0:
        wb[3] = wb[1]
    if wb[1] == 256:
        wb /= wb[1]
    raw = src.astype('float32')
    for c in range(src.shape[2]):
        raw[..., c] = raw[..., c] * wb[c]
    return raw
    
def demosaic_cv2(src, pattern):
    if pattern[1, 1] == 0:
        COLOR_CODE = cv2.COLOR_BAYER_RG2RGB
    elif pattern[1, 1] == 2:
        COLOR_CODE = cv2.COLOR_BAYER_BG2RGB
    return cv2.cvtColor(src, COLOR_CODE)

# # %%
# import numpy as np
# rgbg = np.ones((2,2,4)).astype('uint8')
# h,w = rgbg.shape[:2]
# bayer = np.zeros((h*2, w*2)).astype('uint8')
# print(rgbg.shape, bayer.shape)

# for c in range(rgbg.shape[2]):
#     rgbg[..., c] *= c
#     print(rgbg[:,:,c])
# R, Gr, B, Gb = [(1,1), (1,0), (0,0), (0,1)]

# bayer[R[0]::2, R[1]::2] = rgbg[..., 0]
# bayer[Gr[0]::2,  Gr[1]::2] = rgbg[..., 1]
# bayer[B[0]::2,   B[1]::2] = rgbg[..., 2]
# bayer[Gb[0]::2,  Gb[1]::2] = rgbg[..., 3]
# print(bayer, bayer.shape)

# #%%
# bayer = np.array([[300, 400, 301, 401],
#                   [200, 100, 201, 101],
#                   [310, 410, 311, 411],
#                   [210, 110, 211, 111]])
# print(bayer)
# pattern = np.array([[2,3],
#                     [1,0]])
# rgbg = bayer2rgbg(bayer, pattern)
# print(rgbg.shape)
# for c in range(rgbg.shape[2]):
#     print(rgbg[..., c])

# rebayer = rgbg2bayer(rgbg, pattern)
# print(rebayer)

# #%%
# %%
def preprocess(img, raw, ccm=None, gamma=2.2, toMean=None):
    if len(img.shape) == 3 and img.shape[2] >= 3:
        out = whitebalance_rgb(img, raw.camera_whitebalance)[:,:,:3]
    else:
        out = img
    # out = img
    if type(ccm) == np.ndarray:
        print(out.shape)
        out = color_correct(out, ccm)

    out = out**(1/gamma)
    
    if toMean != None:
        out = out * (toMean/np.mean(out))
    return out

def postprocess(img, raw, toMean=None):
    # # get raw image data
    # image = np.array(raw.raw_image_visible, dtype=np.double)
    # # subtract black levels and normalize to interval [0..1]
    # black = np.reshape(np.array(raw.black_level_per_channel, dtype=np.double), (2, 2))
    # black = np.tile(black, (image.shape[0]//2, image.shape[1]//2))
    # image = (image - black) / (raw.white_level - black)
    # # find the positions of the three (red, green and blue) or four base colors within the Bayer pattern
    # n_colors = raw.num_colors
    # colors = np.frombuffer(raw.color_desc, dtype=np.byte)
    # pattern = np.array(raw.raw_pattern)
    # index_0 = np.where(colors[pattern] == colors[0])
    # index_1 = np.where(colors[pattern] == colors[1])
    # index_2 = np.where(colors[pattern] == colors[2])
    # index_3 = np.where(colors[pattern] == colors[3])
    # # apply white balance, normalize white balance coefficients to the 2nd coefficient, which is ususally the coefficient for green
    # wb_c = raw.camera_whitebalance 
    # wb = np.zeros((2, 2), dtype=np.double) 
    # wb[index_0] = wb_c[0] / wb_c[1]
    # wb[index_1] = wb_c[1] / wb_c[1]
    # wb[index_2] = wb_c[2] / wb_c[1]
    # if n_colors == 4:
    #     wb[index_3] = wb_c[3] / wb_c[1]
    # wb = np.tile(wb, (image.shape[0]//2, image.shape[1]//2))
    # image_wb = np.clip(image * wb, 0, 1)
    # # demosaic via downsampling
    # image_demosaiced = np.empty((image_wb.shape[0]//2, image_wb.shape[1]//2, n_colors))
    # if n_colors == 3:
    #     image_demosaiced[:, :, 0] = image_wb[index_0[0][0]::2, index_0[1][0]::2]
    #     image_demosaiced[:, :, 1]  = (image_wb[index_1[0][0]::2, index_1[1][0]::2] + image_wb[index_1[0][1]::2, index_1[1][1]::2]) / 2
    #     image_demosaiced[:, :, 2]  = image_wb[index_2[0][0]::2, index_2[1][0]::2]
    # else: # n_colors == 4
    #     image_demosaiced[:, :, 0] = image_wb[index_0[0][0]::2, index_0[1][0]::2]
    #     image_demosaiced[:, :, 1] = image_wb[index_1[0][0]::2, index_1[1][0]::2]
    #     image_demosaiced[:, :, 2] = image_wb[index_2[0][0]::2, index_2[1][0]::2]
    #     image_demosaiced[:, :, 3] = image_wb[index_3[0][0]::2, index_3[1][0]::2]
    # # convert to linear sRGB, calculate the matrix that transforms sRGB into the camera's primary color components and invert this matrix to perform the inverse transformation
    # # image_demosaiced = img
    n_colors = raw.num_colors
    image_demosaiced = whitebalance_rgb(img, raw.camera_whitebalance)
    XYZ_to_cam = np.array(raw.rgb_xyz_matrix[0:n_colors, :], dtype=np.double)
    sRGB_to_XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                            [0.2126729, 0.7151522, 0.0721750],
                            [0.0193339, 0.1191920, 0.9503041]], dtype=np.double)
    sRGB_to_cam = np.dot(XYZ_to_cam, sRGB_to_XYZ)
    norm = np.tile(np.sum(sRGB_to_cam, 1), (3, 1)).transpose()
    sRGB_to_cam = sRGB_to_cam / norm
    if n_colors == 3:
        cam_to_sRGB = np.linalg.inv(sRGB_to_cam)
    else: # n_colors == 4
        cam_to_sRGB = np.linalg.pinv(sRGB_to_cam)
    image_sRGB = np.einsum('ij,...j', cam_to_sRGB, image_demosaiced)  # performs the matrix-vector product for each pixel
    # apply sRGB gamma curve
    i = image_sRGB < 0.0031308
    j = np.logical_not(i)
    image_sRGB[i] = 323 / 25 * image_sRGB[i]
    image_sRGB[j] = 211 / 200 * image_sRGB[j] ** (5 / 12) - 11 / 200
    image_sRGB = np.clip(image_sRGB, 0, 1)
    # show image
    # plt.axis('off')
    # plt.imshow(image_sRGB)
    # utils.print_info(image_sRGB)
    if toMean != None:
        image_sRGB = image_sRGB * (toMean/np.mean(image_sRGB))
    return image_sRGB

def cc_match(ours, gt):
    # ours = (ours * 255).astype('uint8')
    # gt = (gt * 255).astype('uint8')
    ours_stdev = np.std(ours, axis=tuple(range(gt.ndim-1)))
    ours_mean = np.mean(ours, axis=tuple(range(gt.ndim-1)))
    ours_mean1 = ours_mean[np.newaxis, np.newaxis, ...]
    ours_stdev1 = ours_stdev[np.newaxis, np.newaxis, ...]

    gt_mean = np.mean(gt, axis=tuple(range(gt.ndim-1)))
    gt_stdev = np.std(gt, axis=tuple(range(gt.ndim-1)))
    gt_mean1 = gt_mean[np.newaxis, np.newaxis, ...]
    gt_stdev1 = gt_stdev[np.newaxis, np.newaxis, ...]

    out = (ours - ours_mean1) * gt_stdev1 / ours_stdev1 + gt_mean1
    out = out[:,:,:3]# / 255
    
    # plt.imshow(out)
    # plt.show()
    return out

def generate_pattern(pattern_type, rows, columns, square_size, page_size):
    """Generate calibration pattern image e.g. Checkerboard

    Args:
        type (str): [description]
        rows (int): [description]
        columns (int): [description]
        square_size (int): [description]
        page_size (tuple/list of int): [description]
    """
    if type(page_size) != list and type(page_size) != tuple:
        page_size = (page_size, page_size)
    spacing = square_size
    
    page = np.ones(page_size).astype('float32')
    border = ((page_size[0] - (rows+1)*square_size)//2,
              (page_size[1] - (columns+1)*square_size)//2)

    for x in range(columns+1):
        for y in range(rows+1):
            if x % 2 == y % 2:
                xstart, xend = [border[1] + x * spacing, 
                                border[1] + x * spacing + spacing]
                ystart, yend = [border[0] + y * spacing, 
                                border[0] + y * spacing + spacing]
                page[ystart:yend, xstart:xend] = 0
    return (np.clip(page, 0, 1) * 255).astype('uint8')

# #%% 
# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# checkerboard = generate_pattern(pattern_type='checkerboard', rows=6, columns=12, square_size=130, page_size=(1080, 1920))
# plt.imshow(checkerboard, cmap='gray'); plt.title('1080x1920, 6x12'); plt.show()
# checkerboard_720 = generate_pattern(pattern_type='checkerboard', rows=6, columns=12, square_size=85, page_size=(720, 1280))
# plt.imshow(checkerboard_720, cmap='gray'); plt.title('1280x720, 6x12'); plt.show()
# checkerboard_720x720 = generate_pattern(pattern_type='checkerboard', rows=6, columns=6, square_size=85, page_size=(720, 720))
# plt.imshow(checkerboard_720x720, cmap='gray'); plt.title('720x720, 6x6'); plt.show()
    
# # %%
# cv2.imwrite('../data/checkerboard/checkerboard_1080x1920_6x12.png', checkerboard)
# cv2.imwrite('../data/checkerboard/checkerboard_720x1280_6x12.png', checkerboard_720)
# cv2.imwrite('../data/checkerboard/checkerboard_720x720_6x6.png', checkerboard_720x720)

# # %%
