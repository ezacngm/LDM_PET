
import numpy as np
import torch
import torch as th
import torch.nn as nn


from torch_radon import ParallelBeam


def addnoise(img,count = 5e6):
    # img = th.from_numpy(img).cuda().float()
    img = (img+1)/2
    if len(img.shape)==4:
        b,c,h,w = img.shape
    elif len(img.shape)==2:
        h,w = img.shape
    elif len(img.shape)==3:
        raise ValueError("input shape is wrong")
    assert h == w ,"input pet image has wrong shape"
    pbeam = ParallelBeam(h,np.linspace(0, np.pi, h, endpoint=False))
    proj = pbeam.forward(img)
    mul_factor = th.ones_like(proj)
    mul_factor = mul_factor + (torch.rand_like(mul_factor) * 0.2 - 0.1)
    noise = torch.ones_like(proj) * torch.mean(mul_factor * proj, dim=(-1, -2), keepdims=True) * 0.2
    sino = mul_factor * proj + noise
    cs = count / (1e-9 + th.sum(sino, dim=(-1, -2), keepdim=True))
    sino = sino * cs
    mul_factor = mul_factor * cs
    noise = noise * cs
    x = th.poisson(sino)
    sinogram = nn.ReLU()((x - noise) / mul_factor)
    # sinogram = pbeam.filter_sinogram(sinogram)
    img = pbeam.backward(sinogram)
    # return sinogram.cpu().numpy().squeeze(),img.cpu().numpy().squeeze()
    return  sinogram
