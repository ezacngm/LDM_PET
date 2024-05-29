import glob
import os
import torch
import torchvision
import numpy as np
from PIL import Image
from utils.diffusion_utils import load_latents
from tqdm import tqdm
from torch.utils.data import Dataset


class PairedDataset(Dataset):
    def __init__(self, split, hq_im_path, lq_im_path, im_size=256, im_channels=3, im_ext='npy',
                 use_latents=False, hq_latent_path=None, lq_latent_path=None, condition_config=None):
        self.split = split
        self.im_size = im_size
        self.im_channels = im_channels
        self.im_ext = im_ext
        self.hq_im_path = hq_im_path
        self.lq_im_path = lq_im_path
        self.hq_latent_maps = None
        self.lq_latent_maps = None
        self.use_latents = False

        self.condition_types = [] if condition_config is None else condition_config['condition_types']

        self.idx_to_cls_map = {}
        self.cls_to_idx_map = {}

        if 'image' in self.condition_types:
            self.mask_channels = condition_config['image_condition_config']['image_condition_input_channels']
            self.mask_h = condition_config['image_condition_config']['image_condition_h']
            self.mask_w = condition_config['image_condition_config']['image_condition_w']

        self.hq_images = self.load_images(hq_im_path)
        self.lq_images = self.load_images(lq_im_path)

        # Ensure both directories have the same number of images
        assert len(self.hq_images) == len(self.lq_images), "Mismatch in dataset lengths"

        # Ensure files are paired correctly
        for hq_file, lq_file in zip(self.hq_images, self.lq_images):
            hq_base = os.path.basename(hq_file).split('.')[0]
            hq_name = [hq_base.split('_')[0], hq_base.split('_')[2]]
            lq_base = os.path.basename(lq_file).split('.')[0]
            lq_name = [lq_base.split('_')[0], lq_base.split('_')[2]]
            assert hq_name == lq_name, f"Mismatched pair: {hq_file}, {lq_file}"

        # Whether to load images or to load latents
        if use_latents:
            if hq_latent_path and lq_latent_path:
                hq_latent_maps = load_latents(hq_latent_path)
                lq_latent_maps = load_latents(lq_latent_path)
                if len(hq_latent_maps) == len(self.hq_images) and len(lq_latent_maps) == len(self.lq_images):
                    self.use_latents = True
                    self.hq_latent_maps = hq_latent_maps
                    self.lq_latent_maps = lq_latent_maps
                    print(f'Found {len(self.hq_latent_maps)} high-quality latents')
                    print(f'Found {len(self.lq_latent_maps)} low-quality latents')
                else:
                    print('Latents not found or mismatched')

    def load_images(self, im_path):
        assert os.path.exists(im_path), f"images path {im_path} does not exist"
        fpaths = glob.glob(os.path.join(im_path, f"*.{self.im_ext}"))
        fpaths.sort()  # Ensure files are loaded in sorted order
        return fpaths

    def normalize_(self, img, min_val=None, max_val=None):
        min_val = img.min()
        max_val = img.max()
        img_normalized = (img - min_val) / (max_val - min_val)
        img_normalized = img_normalized.clip(0, 1)
        return img_normalized

    def __len__(self):
        return len(self.hq_images)

    def __getitem__(self, index):
        if self.use_latents:
            hq_latent = self.hq_latent_maps[self.hq_images[index]]
            lq_latent = self.lq_latent_maps[self.lq_images[index]]
            return hq_latent, lq_latent
        else:
            hq_image = np.load(self.hq_images[index])[np.newaxis, ...]
            lq_image = np.load(self.lq_images[index])[np.newaxis, ...]

            hq_image = torch.from_numpy(hq_image)
            lq_image = torch.from_numpy(lq_image)

            transform = torchvision.transforms.Compose([
                torchvision.transforms.CenterCrop(self.im_size),
            ])

            hq_image = transform(hq_image)
            lq_image = transform(lq_image)

            hq_image = self.normalize_(hq_image)
            lq_image = self.normalize_(lq_image)

            hq_image = (2 * hq_image) - 1
            lq_image = (2 * lq_image) - 1

            if len(self.condition_types) == 0:
                return hq_image, lq_image
            else:
                return hq_image, lq_image


# # Usage example
# hq_im_path = 'path/to/high_quality_images'
# lq_im_path = 'path/to/low_quality_images'
#
# paired_dataset = PairedDataset(split='train', hq_im_path=hq_im_path, lq_im_path=lq_im_path, im_size=256, im_ext='npy')
# data_loader = DataLoader(paired_dataset, batch_size=16, shuffle=True)

# for epoch in range(num_epochs):
#     for hq_images, lq_images in tqdm(data_loader):
#         optimizer.zero_grad()
#
#         hq_images = hq_images.float().to(device)
#         lq_images = lq_images.float().to(device)
#
#         if not paired_dataset.use_latents:
#             with torch.no_grad():
#                 hq_images, _ = vae.encode(hq_images)
#                 lq_images, _ = vae.encode(lq_images)

        # Perform operations with hq_images and lq_images
        # Example: combining images for further processing
        # combined_data = some_function(hq_images, lq_images)

        # Continue with your training steps
        # Example:
        # output = model(combined_data)
        # loss = loss_function(output, target)
        # loss.backward()
        # optimizer.step()

        # Optionally, log the loss or other metrics

    # Optionally, validate your model and log metrics after each epoch
