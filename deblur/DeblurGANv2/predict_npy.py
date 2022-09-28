import os
from glob import glob
from typing import Optional
import time
import cv2
import numpy as np
import torch
import yaml
from fire import Fire
from tqdm import tqdm

from aug import get_normalize
from models.networks import get_generator


class Predictor:
    def __init__(self, weights_path: str, model_name: str = ''):
        with open('config/config.yaml') as cfg:
            config = yaml.load(cfg)
        model = get_generator(model_name or config['model'])
        model.load_state_dict(torch.load(weights_path)['model'])
        self.model = model.cuda()
        self.model.train(True)
        # GAN inference should be in train mode to use actual stats in norm layers,
        # it's not a bug
        self.normalize_fn = get_normalize()

    @staticmethod
    def _array_to_batch(x):
        x = np.transpose(x, (2, 0, 1))
        x = np.expand_dims(x, 0)
        return torch.from_numpy(x)

    def _preprocess(self, x: np.ndarray, mask: Optional[np.ndarray]):
        x, _ = self.normalize_fn(x, x)
        if mask is None:
            mask = np.ones_like(x, dtype=np.float32)
        else:
            mask = np.round(mask.astype('float32') / 255)

        h, w, _ = x.shape
        block_size = 32
        min_height = (h // block_size + 1) * block_size
        min_width = (w // block_size + 1) * block_size

        pad_params = {'mode': 'constant',
                      'constant_values': 0,
                      'pad_width': ((0, min_height - h), (0, min_width - w), (0, 0))
                      }
        x = np.pad(x, **pad_params)
        mask = np.pad(mask, **pad_params)

        return map(self._array_to_batch, (x, mask)), h, w

    @staticmethod
    def _postprocess(x: torch.Tensor) -> np.ndarray:
        x, = x
        x = x.detach().cpu().float().numpy()
        x = ((np.transpose(x, (1, 2, 0)) + 1) / 2.0 ) #* 255.0
        return x#.astype('uint8')

    def __call__(self, img: np.ndarray, mask: Optional[np.ndarray], ignore_mask=True) -> np.ndarray:
        (img, mask), h, w = self._preprocess(img, mask)
        with torch.no_grad():
            inputs = [img.cuda()]
            if not ignore_mask:
                inputs += [mask]
            pred = self.model(*inputs)
        return self._postprocess(pred)[:h, :w, :]

def process_video(pairs, predictor, output_dir):
    for video_filepath, mask in tqdm(pairs):
        video_filename = os.path.basename(video_filepath)
        output_filepath = os.path.join(output_dir, os.path.splitext(video_filename)[0]+'_deblur.mp4')
        video_in = cv2.VideoCapture(video_filepath)
        fps = video_in.get(cv2.CAP_PROP_FPS)
        width = int(video_in.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_in.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frame_num = int(video_in.get(cv2.CAP_PROP_FRAME_COUNT))
        video_out = cv2.VideoWriter(output_filepath, cv2.VideoWriter_fourcc(*'MP4V'), fps, (width, height))
        tqdm.write(f'process {video_filepath} to {output_filepath}, {fps}fps, resolution: {width}x{height}')
        for frame_num in tqdm(range(total_frame_num), desc=video_filename):
            res, img = video_in.read()
            if not res:
                break
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pred = predictor(img, mask)
            pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
            video_out.write(pred)


def center_crop(img, crop):
    if type(crop) != list and type(crop) != tuple:
        crop = [crop, crop]
    h, w = img.shape[:2]
    return img[h//2-crop[0]//2:h//2+crop[0]//2, w//2-crop[1]//2:w//2+crop[1]//2]

def main(img_pattern: str='./images/real/main/*.npy',
         mask_pattern: Optional[str] = None,
         weights_path= './pretrained/best_fpn.h5', 
         out_dir='submit/best_fpn/awb/',
         side_by_side: bool = False,
         video: bool = False):
    def sorted_glob(pattern):
        return sorted(glob(pattern))
    os.makedirs(out_dir + 'npy/', exist_ok=True)
    os.makedirs(out_dir + 'jpg/', exist_ok=True)
    
    imgs = sorted_glob(img_pattern)
    masks = sorted_glob(mask_pattern) if mask_pattern is not None else [None for _ in imgs]
    pairs = zip(imgs, masks)
    names = sorted([os.path.basename(x) for x in glob(img_pattern)])
    predictor = Predictor(weights_path=weights_path)

    os.makedirs(out_dir, exist_ok=True)
    if not video:
        for name, pair in tqdm(zip(names, pairs), total=len(names)):
            f_img, f_mask = pair
            img, mask = [np.load(f_img).astype('float32'), f_mask]

            h_in, w_in = img.shape[:2]
            h, w = img.shape[:2]
            print(h, w)
            
            cv2.imwrite(os.path.join(out_dir + 'jpg/', name[:-4]+'_in.jpg'),
                        np.uint8(np.clip(img, 0, 1)*255)[:,:,::-1])
            
            start = time.time()
            pred = np.zeros_like(img, dtype=np.float32)
            grid = (1,1)
            
            for i in range(grid[0]):
                for j in range(grid[1]):
                    overlapH = 0 if i==0 else 100
                    overlapW = 0 if j==0 else 100
                    tmp = img[i*h//grid[0]-overlapH:i*h//grid[0]+h//grid[0], j*w//grid[1]-overlapW:j*w//grid[1]+w//grid[1]]
                    
                    tmp_pred = predictor(tmp*255, mask)
                    pred[i*h//grid[0]-overlapH:i*h//grid[0]+h//grid[0], j*w//grid[1]-overlapW:j*w//grid[1]+w//grid[1]] = tmp_pred
            
            print(time.time() - start)

            np.save(out_dir + 'npy/' + name[:-4] + '_deblur.npy', pred)
            if side_by_side:
                pred = np.hstack((img, pred))
       
            cv2.imwrite(os.path.join(out_dir + 'jpg/', name[:-4]+'_deblur.jpg'),
                        np.uint8(np.clip(pred, 0, 1)*255)[:,:,::-1])
            torch.cuda.empty_cache()
    else:
        process_video(pairs, predictor, out_dir)


if __name__ == '__main__':
    Fire(main)

