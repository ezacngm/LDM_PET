import torch
import yaml
import argparse
import os
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
from dataset.mnist_dataset import MnistDataset
from dataset.celeb_dataset import CelebDataset
from dataset.pet_dataset import PetDataset
from dataset.paired_dataset import PairedDataset
from torch.utils.data import DataLoader
from models.unet_base import Unet
from models.vqvae import VQVAE
from scheduler.linear_noise_scheduler import LinearNoiseScheduler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(args):
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
    
    im_dataset_cls = {
        'mnist': MnistDataset,
        'celebhq': CelebDataset,
        'ppet': PetDataset,
    }.get(dataset_config['name'])
    
    im_dataset = im_dataset_cls(split='train',
                                im_path=dataset_config['im_path'],
                                im_size=dataset_config['im_size'],
                                im_channels=dataset_config['im_channels'],
                                use_latents=True,
                                latent_path=os.path.join(train_config['task_name'],
                                                         train_config['vqvae_latent_dir_name'])
                                )



    # data_loader = DataLoader(im_dataset,
    #                          batch_size=train_config['ldm_batch_size'],
    #                          shuffle=True)


    paired_dataset = PairedDataset(split='train', hq_im_path=dataset_config['im_path'],
                                   lq_im_path=dataset_config['low_im_path'],
                                   im_size=dataset_config['im_size'],
                                   im_channels=dataset_config['im_channels'],
                                   use_latents=False,
                                   )
    data_loader = DataLoader(paired_dataset, batch_size=train_config['ldm_batch_size'], shuffle=True)

    # Instantiate the model
    model = Unet(im_channels=autoencoder_model_config['z_channels']*2,
                 model_config=diffusion_model_config).to(device)
    model.train()
    
    # Load VAE ONLY if latents are not to be used or are missing
    if not im_dataset.use_latents:
        print('Loading vqvae model as latents not present')
        vae = VQVAE(im_channels=dataset_config['im_channels'],
                    model_config=autoencoder_model_config).to(device)
        vae.eval()
        # Load vae if found
        if os.path.exists(os.path.join(train_config['task_name'],
                                       train_config['vqvae_autoencoder_ckpt_name'])):
            print('Loaded vae checkpoint')
            vae.load_state_dict(torch.load(os.path.join(train_config['task_name'],
                                                        train_config['vqvae_autoencoder_ckpt_name']),
                                           map_location=device))
    # Specify training parameters
    num_epochs = train_config['ldm_epochs']
    optimizer = Adam(model.parameters(), lr=train_config['ldm_lr'])
    criterion = torch.nn.MSELoss()
    
    # Run training
    if not im_dataset.use_latents:
        for param in vae.parameters():
            param.requires_grad = False

    for epoch_idx in range(num_epochs):
        losses = []
        progress_bar = tqdm(data_loader, desc='train', leave=True, position=0)
        for im,im_low in progress_bar:
            optimizer.zero_grad()
            im = im.float().to(device)
            im_low = im_low.float().to(device)
            if not im_dataset.use_latents:
                with torch.no_grad():
                    im, _ = vae.encode(im)
                    print("im is latent encoded")
                    im_low, _ = vae.encode(im_low)
            
            # Sample random noise
            noise = torch.randn_like(im).to(device)
            
            # Sample timestep
            t = torch.randint(0, diffusion_config['num_timesteps'], (im.shape[0],)).to(device)
            
            # Add noise to images according to timestep
            noisy_im = scheduler.add_noise(im, noise, t)
            noise_pred = model(torch.cat([noisy_im,im_low],dim=1), t)
            
            loss = criterion(noise_pred, noise)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

            progress_bar.set_postfix(
                loss=np.mean(losses)
            )

        print('Finished epoch:{} | Loss : {:.4f}'.format(
            epoch_idx + 1,
            np.mean(losses)))
        
        torch.save(model.state_dict(), os.path.join(train_config['task_name'],
                                                    train_config['ldm_ckpt_name']))
    
    print('Done Training ...')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for ddpm training')
    parser.add_argument('--config', dest='config_path',
                        default='config/mnist.yaml', type=str)
    args = parser.parse_args()
    train(args)
