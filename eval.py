import math
import numpy as np
from PIL import Image
import lpips
import torch

def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))
import math
import numpy as np
import cv2

def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


# def reshape

# img1 = Image.open("pixel/preprocessed_image.png")
# # img1 = Image.open("examples/kunkun.webp")
# img2 = Image.open("pixel/refrence_image.png")
# new_size = (256, 256)
# img1 = img1.resize(new_size)
# img1 = np.array(img1)

# img2 = np.array(img2)
# # print(img1.shape, img2.shape)
# # print(calculate_psnr(img1, img2))
# print(calculate_ssim(img1, img2))

def calculate_lpips(img1, img2,loss_fn_vgg):
    # loss_fn_vgg = lpips.LPIPS(net="vgg")
    img1 = torch.tensor(np.array(img1)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img1 = img1 * 2.0 - 1.0
    img2 = torch.tensor(np.array(img2)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img2 = img2 * 2.0 - 1.0
    return loss_fn_vgg(img1, img2).item()

def calculate_metrics(img1, img2,loss_pips):
    size = (256, 256)
    img1 = img1.resize(size)
    img1 = np.array(img1)   
    img2 = np.array(img2)
    psnr = calculate_psnr(img1, img2)
    ssim = calculate_ssim(img1, img2)
    lpips = calculate_lpips(img1, img2,loss_pips)
    return psnr, ssim, lpips
    # print(calculate_lpips(img1, img2))
    # return calculate_psnr(img1, img2), calculate_ssim(img1, img2)

# print(calculate_metrics(img1, img2,lpips.LPIPS(net="vgg")) )