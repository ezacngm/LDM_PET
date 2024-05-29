import glob
import os
import random
import torch
import torchvision
import numpy as np
from PIL import Image
from utils.diffusion_utils import load_latents
from tqdm import tqdm
from torch.utils.data.dataset import Dataset


class PetDataset(Dataset):
    r"""
    Celeb dataset will by default centre crop and resize the images.
    This can be replaced by any other dataset. As long as all the images
    are under one directory.
    """
    
    def __init__(self, split, im_path, im_size=256, im_channels=3, im_ext='npy',
                 use_latents=False, latent_path=None, condition_config=None):
        self.split = split
        self.im_size = im_size
        self.im_channels = im_channels
        self.im_ext = im_ext
        self.im_path = im_path
        self.latent_maps = None
        self.use_latents = False
        
        self.condition_types = [] if condition_config is None else condition_config['condition_types']
        
        self.idx_to_cls_map = {}
        self.cls_to_idx_map ={}
        
        if 'image' in self.condition_types:
            self.mask_channels = condition_config['image_condition_config']['image_condition_input_channels']
            self.mask_h = condition_config['image_condition_config']['image_condition_h']
            self.mask_w = condition_config['image_condition_config']['image_condition_w']
            
        self.images = self.load_images(im_path)
        
        # Whether to load images or to load latents
        if use_latents and latent_path is not None:
            latent_maps = load_latents(latent_path)
            if len(latent_maps) == len(self.images):
                self.use_latents = True
                self.latent_maps = latent_maps
                print('Found {} latents'.format(len(self.latent_maps)))
            else:
                print('Latents not found')
    
    def load_images(self, im_path):
        r"""
        Gets all images from the path specified
        and stacks them all up
        """
        assert os.path.exists(im_path), "images path {} does not exist".format(im_path)
        fpaths = glob.glob(os.path.join(im_path, "*.{}".format('npy')))
        fold = 0.1
        if self.split == "train":
            f_final = fpaths[int(fold*len(fpaths)):]
        else:
            f_final = fpaths[:int(fold*len(fpaths))]
        return f_final

    def normalize_(self, img, min_val=None, max_val=None):
        min_val = img.min()
        max_val = img.max()
        # Normalize PET images to a [0, 1] range
        img_normalized = (img - min_val) / (max_val - min_val)
        # Clip values just in case to ensure they remain within [0, 1]
        img_normalized = img_normalized.clip(0, 1)
        # img = transforms.RandomRotation(degrees=(-15, 15))(img)  # todo: 看torch文档 要求输入shape是 BCWH
        # img = transforms.RandomHorizontalFlip(p=0.5)(img)
        return img_normalized
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        if self.use_latents:
            latent = self.latent_maps[self.images[index]]
            if len(self.condition_types) == 0:
                return latent
            else:
                return latent
        else:
            im = np.load(self.images[index])[np.newaxis, ...]
            im = torch.from_numpy(im)
            im_tensor = torchvision.transforms.Compose([
                torchvision.transforms.CenterCrop(self.im_size),
            ])(im)
            im_tensor = self.normalize_(im_tensor)
            # Convert input to -1 to 1 range.
            im_tensor = (2 * im_tensor) - 1
            if len(self.condition_types) == 0:
                return im_tensor
            else:
                return im_tensor
