# train

## first train vae
python tools/train_vqvae.py --config config/ppet.yaml
## then train ldm
python tools/train_ddpm_vqvae.py --config config/ppet.yaml

# sample
python tools/sample_ddpm_vqvae.py --config config/ppet.yaml

# for changing to arbitrary dose level
change the configuration file's argument, **im_path** ,to any folders that contain PET images with different dose-level.
