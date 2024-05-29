import math
import random

from PIL import Image
import blobfile as bf
import numpy as np
import mpi4py
from math import exp
import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import torch.nn.functional as F
import time
from torch_radon import ParallelBeam
import torch.nn as nn

from PIL import Image
from torch.autograd import Variable


def SSIM_tensor(a,b):
    n = a.shape[0]
    data_range = 255
    ssims=[]
    for i in range(n):
        img1 = a[i].cpu().numpy().squeeze()
        img2 = b[i].cpu().numpy().squeeze()
        img1 = data_range*(img1-img1.min())/(img1.max()-img1.min())
        img2 = data_range*(img2-img2.min())/(img2.max()-img1.min())
        ssim_single = compute_SSIM(img1, img2,data_range=data_range)
        ssims.append(ssim_single)
    print("ssimMean",np.mean(ssims))
    # return np.mean(ssims),ssims






def compute_SSIM(img1, img2, data_range, window_size=11, channel=1, size_average=True):
    # referred from https://github.com/Po-Hsun-Su/pytorch-ssim
    if len(img1.shape) == 2:
        h, w = img1.shape
        if type(img1) == torch.Tensor:
            img1 = img1.view(1, 1, h, w)
            img2 = img2.view(1, 1, h, w)
        else:
            img1 = torch.from_numpy(img1[np.newaxis, np.newaxis, :, :]).float()
            img2 = torch.from_numpy(img2[np.newaxis, np.newaxis, :, :]).float()
    window = create_window(window_size, channel)
    window = window.type_as(img1)

    mu1 = F.conv2d(img1, window, padding=window_size // 2)
    mu2 = F.conv2d(img2, window, padding=window_size // 2)
    mu1_sq, mu2_sq = mu1.pow(2), mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2) - mu1_mu2

    C1, C2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    # C1, C2 = 0.01**2, 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    # if size_average:
    #     return ssim_map.mean().item()
    # else:
    #     return ssim_map.mean(1).mean(1).mean(1).item()
    if size_average:
        return ssim_map.mean().item()
    else:
        return ssim_map.mean(1).mean(1).mean(1).item()


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window