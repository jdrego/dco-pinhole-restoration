import os
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import rawpy
import math

def normalize_minmax(img):
    return (img - img.min()) / (img.max() - img.min())

def convert_range(array, in_range, out_range):
    array = array.astype('float64')
    array = (array - in_range[0]) / (in_range[1] - in_range[0]) # 0-1: (x-oldMin)/(oldMax-oldMin)
    return array * (out_range[1] - out_range[0]) + out_range[0] # x*(newMax-newMin) + newMin

def makedirs(paths, exist_ok=True):
    if type(paths) != list:
        paths = [paths]
    for path in paths:
        os.makedirs(path, exist_ok=exist_ok)
        
def imshow(imgs, titles=None, main_title=None, grid_dim=None, dpi=None, axis=False, save=None):
    if type(imgs) is not list:
        imgs = [imgs]
    if type(titles) is not list:
        titles = [titles] if titles != None else [None]
    if grid_dim is None:
        grid_dim = (1, len(imgs))
        
    if dpi is not None:
        plt.figure(dpi=dpi)
        
    for i in range(len(imgs)):
        plt.subplot(grid_dim[0], grid_dim[1], i + 1)
        plt.imshow(cp.asnumpy(imgs[i]), cmap='gray')
        if axis == False:
            plt.axis('off')
        if titles != [None]:
            plt.title(titles[i])
            
    if main_title is not None:
        plt.suptitle(main_title)
    plt.tight_layout()
    if save is not None:
        plt.savefig(save, bbox_inches='tight')
    plt.show()

def plot(XYpair, title=None, yscale=None, main_title=None, grid_dim=None, dpi=None, save=None):
    """
    Plot 2D graph given single (X,Y) pair or subplots with multiple (X,Y) pairs

    Args:
        XYpair (tuple/list of tuples): Pair of (X,Y) axis data to be plotted.
                i.e. XYpair=(X,Y) or XYpair=[(x1,y1), (x2,y2), ...] 
            
        title (str/list of str): (Optional) Title for each (X,Y) subplot
        
        yscale (str): (Optional) Scale the Y-axis
                Options: 'log', ... TODO: Add other options
                
        main_title (str): (Optional) Super title for figure
        
        grid_dim (int, int): (Optional) Grid dimensions for subplots in figure.
                Default: (1, len(XYpair)) TODO: rewrite this better
        
        dpi (int): (Optional) DPI for figure
        
        save (str): (Optional) Path and filename to save figure

    Returns: None
    """
    if type(XYpair) is not list:
        XYpair = [XYpair]
    if type(title) is not list:
        title = [title] if title != None else [None]
    if type(yscale) is not list:
        yscale = [yscale]
    if grid_dim is None:
        grid_dim = (1, len(XYpair))
        
    if dpi is not None:
        plt.figure(dpi=dpi)
        
    for i in range(len(XYpair)):
        plt.subplot(grid_dim[0], grid_dim[1], i + 1)
        plt.plot(XYpair[i][0], XYpair[i][1])
        if yscale != [None]:
            plt.yscale(yscale[i])
        # if axis == False:
        #     plt.axis('off')
        if title != [None]:
            plt.title(title[i])
            
    if main_title is not None:
        plt.suptitle(main_title)
    plt.tight_layout()
    if save is not None:
        plt.savefig(save, bbox_inches='tight')
    plt.show()
    
def print_info(array, title=None):
    if type(array) is not list:
        array = [array]
    if type(title) is not list:
        title = [title] if title != None else [None]

    for i in range(len(array)):
        title_i = title[i] if title != [None] else ''  
        print(title_i, 'DTYPE:', array[i].dtype, 'SHAPE', array[i].shape, 
              'MIN:', array[i].min(), 'MAX:', array[i].max())

def print_raw_info(raw):
    print('* raw.raw_type\t:\t',
          '"Return the RAW type."\n', 
          raw.raw_type)
    sz = raw.sizes
    print('* raw.sizes\t:\t',
          '"rawpy.ImageSizes instance w/ size info of the RAW & processed image."\n', 
          '{}\t=\t{}\n'.format(sz._fields[:2], sz[:2]), '{}\t\t=\t{}\n'.format(sz._fields[2:4], sz[2:4]),
          '{}\t=\t{}\n'.format(sz._fields[4:6], sz[4:6]), '{}\t\t=\t{}\n'.format(sz._fields[6:8], sz[6:8]),
          '{}\t=\t{}'.format(sz._fields[8:10], sz[8:10]))
    print()
    print('* raw.color_desc\t:\t',
          '"String descr. of colors numbered from 0 to 3 (RGBG,RGBE,GMCY, or GBTG)."\n', 
          raw.color_desc)
    print('* raw.num_colors\t:\t',
          '"Number of colors. For RGBG it can be 3 or 4, depends on the cam model."\n', 
          raw.num_colors)
    print()
    print('* raw.black_level_per_channel :\t',
          '"Per-channel black level correction."\n', 
          raw.black_level_per_channel)
    print('* raw.white_level\t:\t',
          '"Level at which the raw pixel value is considered to be saturated."\n', 
          raw.white_level)
    print('* RAW bit-depth \t:\t',
          '"Manually converted from white_level"\n',
          math.log2(raw.white_level+1), 'bit')
    print()
    print('* raw.raw_pattern\t:\t',
          '"The smallest possible Bayer pattern of this image."\n', 
          raw.raw_pattern)
    print('* raw.raw_colors\t:\t',
          '"(Top-left) An array of color indices for each pixel in the RAW image."\n', 
          raw.raw_colors[:2,:2])
    print('* raw.raw_colors_visible:\t',
          '"(Top-left) Like raw_colors but without margin."\n', 
          raw.raw_colors_visible[:2,:2])
    print('* raw.color_matrix\t:\t',
          '"Color matrix, read from file for some cameras, calculated for others."\n', 
          raw.color_matrix)
    print()
    print('* raw.camera_whitebalance\t: ',
          '"White balance coeffs (as shot). Either read from file or calculated."\n', 
          raw.camera_whitebalance)
    print('* raw.daylight_whitebalance\t: ',
          '"White balance coefficients for daylight (daylight balance)."\n', 
          raw.daylight_whitebalance)
    print()
    print('* raw.rgb_xyz_matrix\t:\t',
          '"Camera RGB-XYZ conversion matrix. This matrix is constant."\n', 
          raw.rgb_xyz_matrix)
    print()
    print('Additional available info for use:')
    print('\t* raw.raw_image \t:\t "View of RAW image. Includes margin"')
    print('\t* raw.raw_image_visible\t:\t "Like raw_image but without margin"')
    print('\t* raw.tone_curve \t:\t "Camera tone curve, read from file."')
    print()
    
def pack_raw(bayer, pattern):
    """ Pack single channel Bayer RAW image into RGrBGb 4 channel array

    Args:
        bayer (np.array): Single channel bayer image of shape (H, W)
        pattern (np.array): Raw Bayer 2x2 pattern with numbers corresponding to R Gr B Gb order.
                            e.g. For R Gr B Gb -> pattern = [[0, 1]]
                                                             [2] 

    Returns:
        [type]: [description]
    """
    # pack Bayer image to 4 channels
    # im = raw.raw_image_visible.astype(np.float32)
    # im = np.maximum(im - 512, 0) / (16383 - 512)  # subtract the black level

    im = np.expand_dims(bayer, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    
    # RGBG
    R = np.argwhere(pattern == 0)[0]
    Gr = np.argwhere(pattern == 1)[0]
    B = np.argwhere(pattern == 2)[0]
    Gb = np.argwhere(pattern == 3)[0]

    # out = np.concatenate((im[R[0]:H:2, R[1]:W:2, :],
    #                       im[Gr[0]:H:2, Gr[1]:W:2, :],
    #                       im[B[0]:H:2, B[1]:W:2, :],
    #                       im[Gb[0]:H:2, Gb[1]:W:2, :]), axis=2)
    out = np.concatenate((im[R[0]:H:2, R[1]:W:2, :],
                          (im[Gr[0]:H:2, Gr[1]:W:2, :] + im[Gb[0]:H:2, Gb[1]:W:2, :]) / 2,
                          im[B[0]:H:2, B[1]:W:2, :]), axis=2)
   
    return out

def scale_levels(img, black_level, white_level):        
    if len(img.shape) is 2:
        img = np.expand_dims(img, axis=2)
    img = img.astype('float32')
   
    for c in range(img.shape[2]):
        img[:, :, c] = np.maximum(img[:, :, c] - black_level[c], 0) / (white_level - black_level[c])
    return np.squeeze(img)
    
def match_dim(data, dim):
    """ Resize image dimensions using crop or padding instead of up/down-sampling.

    Args:
        data (np.array): single channel or 3 channel image array. 
            If 3-channel array (i.e. RGB), dimension -1 should be channels.
        dim (int, int): Desired dimensions for H, W. i.e dim = (H, W)

    Returns:
        np.array: Input image matched to new dimensions.
    """
    # Pad outer regions of detector
    if data.shape[0] < dim[0] or data.shape[1] < dim[1]:
        data = pad_edges(data, dim[:2])
    # Crop out edge regions outside detector dimensions     
    if data.shape[0] > dim[0] or data.shape[1] > dim[1]:
        data = center_crop(data, dim[:2])
    return data

def pad_edges(data, dim):
    """ Pads H, W dimensions on outer edges to match desired dimensions if input dimension is lesser.
        If difference between input and output dimension shape is odd, 
        an additional pixel is added on the bottom/right padding.

    Args:
        data (np.array): single channel or 3 channel (i.e. RGB) image array. 
            If 3-channel array, dimension -1 should be channels
        dim (int, int): Desired dimensions for H, W. i.e dim = (H, W)

    Returns:
        np.array: Input image matched to new dimensions.
    """
    pad_h, pad_w = [max(dim[0] - data.shape[0], 0), 
                    max(dim[1] - data.shape[1], 0)]
    pad_top = pad_bot = pad_h // 2
    pad_left = pad_right = pad_w // 2
    
    if pad_h % 2 != 0:
        pad_bot += 1
    if pad_w % 2 != 0:
        pad_right += 1
    pad_tuple = ((pad_top, pad_bot), (pad_left, pad_right))
    if len(data.shape) == 3:
        pad_tuple = pad_tuple + ((0, 0),)
    return np.pad(data, pad_width=pad_tuple)

def center_crop(data, dim):
    """ Crops center H, W dimensions to match desired dimensions if input dimension is greater.

    Args:
        data (np.array): single channel or 3-channel image array. 
        dim (int, int): Desired dimensions for H, W. i.e dim = (H, W)

    Returns:
        np.array: Input image matched to new dimensions.
    """
    if type(dim) == int:
        dim = [dim, dim]
        
    h_start, w_start = [max(data.shape[0] - dim[0], 0) // 2,
                        max(data.shape[1] - dim[1], 0) // 2]
    h_end, w_end = [h_start + min(dim[0], data.shape[0]),
                    w_start + min(dim[1], data.shape[1])]
    return data[h_start:h_end, w_start:w_end]



def hist_plot(img): 
    """ function to obtain histogram of an image 

    Args:
        img (np.array): Single channel 2D image.
    """
    m, n = img.shape[:2]
    
    # empty list to store the count of each intensity value
    count =[] 
    # empty list to store intensity value
    r = [] 
      
    # loop to traverse each intensity value
    for k in range(0, 256): 
        r.append(k) 
        count1 = 0
          
        # loops to traverse each pixel in the image
        for i in range(m): 
            for j in range(n): 
                if img[i, j]== k: 
                    count1+= 1
        count.append(count1) 
    
    # plotting the histogram 
    plt.stem(r, count) 
    plt.xlabel('intensity value') 
    plt.ylabel('number of pixels') 
    plt.title('Histogram of the stretched image') 
    plt.show()
    
    # return (r, count) 