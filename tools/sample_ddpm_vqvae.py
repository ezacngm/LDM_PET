import torch
import torchvision
import argparse
import yaml
import os
import numpy as np
from torchvision.utils import make_grid
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import DataLoader
from dataset.celeb_dataset import CelebDataset
from dataset.mnist_dataset import MnistDataset
from dataset.pet_dataset import PetDataset
from models.unet_base import Unet
from models.vqvae import VQVAE
from scheduler.linear_noise_scheduler import LinearNoiseScheduler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def sample(model, scheduler, train_config, diffusion_model_config,
               autoencoder_model_config, diffusion_config, dataset_config, vae):
    r"""
    Sample stepwise by going backward one timestep at a time.
    We save the x0 predictions
    """
# added by mu nan -----------------

    # Create the dataset
    im_dataset_cls = {
        'mnist': MnistDataset,
        'celebhq': CelebDataset,
        'ppet': PetDataset,
    }.get(dataset_config['name'])
    im_dataset = im_dataset_cls(split='train',
                                im_path=dataset_config['low_im_path'],
                                im_size=dataset_config['im_size'],
                                im_channels=dataset_config['im_channels'])

    gt_dataset = im_dataset_cls(split='train',
                                im_path=dataset_config['im_path'],
                                im_size=dataset_config['im_size'],
                                im_channels=dataset_config['im_channels'])

    # This is only used for saving latents. Which as of now
    # is not done in batches hence batch size 1
    data_loader = DataLoader(im_dataset,
                             batch_size=1,
                             shuffle=False)

    num_images = train_config['num_samples']
    ngrid = train_config['num_grid_rows']
    ims_nms = []
    ims_nms_gt = []
    for i in range(num_images):
        imim = im_dataset[i+200][np.newaxis,]
        im_gt = gt_dataset[i+200][np.newaxis,]
        imim = imim.to(device)
        im_gt = im_gt.to(device)
        ims_nms.append(imim)
        ims_nms_gt.append(im_gt)
    # idxs = torch.randint(0, len(im_dataset) - 1, (num_images,))
    # ims_nm = torch.cat([im_dataset[idx][None, :] for idx in idxs]).float()

    ims_nm = torch.cat(ims_nms,dim=0).float()
    ims_nms_gt = torch.cat(ims_nms_gt,0).float()
    ims_nm = ims_nm.to(device)
    ims_nms_gt = ims_nms_gt.to(device)

    ims_nms_gt = torch.clamp(ims_nms_gt, -1., 1.).detach().cpu()
    ims_nms_gt = (ims_nms_gt + 1) / 2
    grid = make_grid(ims_nms_gt, nrow=train_config['num_grid_rows'])
    img_gt = torchvision.transforms.ToPILImage()(grid)

    if not os.path.exists(os.path.join(train_config['task_name'], 'samples')):
        os.mkdir(os.path.join(train_config['task_name'], 'samples'))
    img_gt.save(os.path.join(train_config['task_name'], 'samples', 'x0_gt.png'))
    img_gt.close()
# ended added by mu nan -------------


    im_size = dataset_config['im_size'] // 2**sum(autoencoder_model_config['down_sample'])
    xt = torch.randn((train_config['num_samples'],
                      autoencoder_model_config['z_channels'],
                      im_size,
                      im_size)).to(device)
    low_cond,_ = vae.encode(ims_nm)
    # low_cond = torch.clamp(low_cond, -1., 1.)
    # low_cond = (low_cond + 1) / 2

    save_count = 0
    for i in tqdm(reversed(range(diffusion_config['num_timesteps']))):
        # Get prediction of noise
        noise_pred = model(torch.cat([xt,low_cond],1), torch.as_tensor(i).unsqueeze(0).to(device))
        
        # Use scheduler to get x0 and xt-1
        xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i).to(device))

        # Save x0
        #ims = torch.clamp(xt, -1., 1.).detach().cpu()
        # if i == 0:
        if i % 50 == 0 or i == diffusion_config['num_timesteps']-1:
            # Decode ONLY the final iamge to save time
            ims = vae.decode(xt)
        # else:
        #     ims = xt
        # ims = vae.decode(xt)

            ims = torch.clamp(ims, -1., 1.).detach().cpu()
            ims = (ims + 1) / 2
            grid = make_grid(ims, nrow=train_config['num_grid_rows'])
            img = torchvision.transforms.ToPILImage()(grid)

            if not os.path.exists(os.path.join(train_config['task_name'], 'samples')):
                os.mkdir(os.path.join(train_config['task_name'], 'samples'))
            img.save(os.path.join(train_config['task_name'], 'samples', 'x0_{}.png'.format(i)))
            img.close()
        if i ==0:
            delta = ims_nms_gt - ims
            # vmin = delta.min().item()
            # vmax = delta.max().item()
            #
            # # Create a grid of images
            # grid = make_grid(delta, nrow=train_config['num_grid_rows'])
            #
            # # Convert the grid to a NumPy array
            # grid_np = grid.numpy().transpose((1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
            #
            # # Normalize the grid values to fit the [0, 1] range if needed
            # grid_np = (grid_np - vmin) / (vmax - vmin)
            #
            # # Create and save the image with the 'jet' colormap
            # fig, ax = plt.subplots()
            # cax = ax.imshow(grid_np, cmap='jet', vmin=0, vmax=1)
            # ax.axis('off')  # Hide axes
            #
            # if not os.path.exists(os.path.join(train_config['task_name'], 'samples')):
            #     os.makedirs(os.path.join(train_config['task_name'], 'samples'))
            #
            # # Save the image
            # fig.savefig(os.path.join(train_config['task_name'], 'samples', 'x0_delta.png'), bbox_inches='tight',
            #             pad_inches=0)
            # plt.close(fig)

            vmin = delta.min().item()
            vmax = delta.max().item()

            # Create a grid of images
            grid_delta = make_grid(delta, nrow=train_config['num_grid_rows'])

            # Convert the grid to a NumPy array
            grid_delta_np = grid_delta.numpy().transpose((1, 2, 0))  # (H, W, C)

            # Normalize the delta grid values for colormap
            grid_delta_np_norm = (grid_delta_np - vmin) / (vmax - vmin)

            # Plot and save the image with the false color map
            fig, ax = plt.subplots()
            cax = ax.imshow(grid_delta_np_norm, cmap='jet', vmin=0, vmax=1)
            fig.colorbar(cax, ax=ax, orientation='vertical')
            ax.axis('off')

            if not os.path.exists(os.path.join(train_config['task_name'], 'samples')):
                os.makedirs(os.path.join(train_config['task_name'], 'samples'))

            fig.savefig(os.path.join(train_config['task_name'], 'samples', 'x0_false_color.png'), bbox_inches='tight',
                        pad_inches=0)
            plt.close(fig)



def infer(args):
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    ########################
    
    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    diffusion_model_config = config['ldm_params']
    autoencoder_model_config = config['autoencoder_params']
    train_config = config['train_params']
    
    # Create the noise scheduler
    scheduler = LinearNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'],
                                     beta_start=diffusion_config['beta_start'],
                                     beta_end=diffusion_config['beta_end'])
    
    model = Unet(im_channels=autoencoder_model_config['z_channels']*2,
                 model_config=diffusion_model_config).to(device)
    model.eval()
    if os.path.exists(os.path.join(train_config['task_name'],
                                   train_config['ldm_ckpt_name'])):
        print('Loaded unet checkpoint')
        model.load_state_dict(torch.load(os.path.join(train_config['task_name'],
                                                      train_config['ldm_ckpt_name']),
                                         map_location=device))
    # Create output directories
    if not os.path.exists(train_config['task_name']):
        os.mkdir(train_config['task_name'])
    
    vae = VQVAE(im_channels=dataset_config['im_channels'],
                model_config=autoencoder_model_config).to(device)
    vae.eval()
    
    # Load vae if found
    if os.path.exists(os.path.join(train_config['task_name'],
                                                    train_config['vqvae_autoencoder_ckpt_name'])):
        print('Loaded vae checkpoint')
        vae.load_state_dict(torch.load(os.path.join(train_config['task_name'],
                                                    train_config['vqvae_autoencoder_ckpt_name']),
                                       map_location=device), strict=True)
    with torch.no_grad():
        sample(model, scheduler, train_config, diffusion_model_config,
               autoencoder_model_config, diffusion_config, dataset_config, vae)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for ddpm image generation')
    parser.add_argument('--config', dest='config_path',
                        default='config/mnist.yaml', type=str)
    args = parser.parse_args()
    infer(args)
