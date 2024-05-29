import numpy as np
import torch
import torch as th
import torch.nn as nn


from torch_radon import ParallelBeam



def performMLEM_radon(image, data, iters=50):
    """
    Perform MLEM using radon transform for PET image reconstruction.

    Parameters:
    image : 2D numpy array
        Initial guess for the image.
    data : 2D numpy array
        Sinogram data (projection data).
    theta : 1D numpy array
        Projection angles (in degrees).
    iters : int
        Number of iterations.

    Returns:
    guess : 2D numpy array
        Reconstructed image.
    """
    data = data.cuda()
    image = image.cuda()
    if len(image.shape)==4:
        b,c,h,w = image.shape
    elif len(image.shape)==2:
        h,w = image.shape
    elif len(image.shape)==3:
        raise ValueError("input shape is wrong")
    assert h == w ,"input pet image has wrong shape"
    pbeam = ParallelBeam(h,np.linspace(0, np.pi, h, endpoint=False))
    guess = torch.ones_like(image)
    ones = torch.ones_like(image)
    for i in range(iters):
        # Forward projection (radon transform)
        projection = pbeam.forward(guess)

        # Ratio of measured data to projection
        ratio = data / (projection + 1e-10)

        # Backprojection (filtered backprojection)
        backprojection = pbeam.backward(ratio)

        # Sensitivity (backprojection of ones)
        sensitivity = pbeam.backward(ones)

        # Update guess
        guess = guess * (backprojection / (sensitivity + 1e-10))

        # Optional: print or log progress
        # print(np.linalg.norm(radon(guess, theta=theta, circle=True) - data))
        # print(i)

    return guess
