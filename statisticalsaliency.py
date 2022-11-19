from numba import jit 
import numpy as np
import cv2 as cv
import numpy as np
import skimage as ski


@jit(nopython=True)
def pixel_saliency(value, hist):
    counter = 0
    for pixel_value, frequency in enumerate(hist):
            counter += abs(value - pixel_value) * frequency
    return counter

@jit(nopython=True)
def single_channel_saliency(im, hist):
    saliency = np.zeros(im.shape)
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            saliency[i,j] = pixel_saliency(im[i,j], hist)
    return saliency

#@jit(nopython=False)
def statistical_saliency(im, wl = 1./3, wa = 1./3, wb = 1./3, sigma = 1):
    im_lab = ski.color.rgb2lab(im)
    saliency = np.zeros(im.shape)
    for c in range(im.shape[2]):
        hist, bin_edges = np.histogram(im_lab[:,:,c], bins=256, range=(0,1))
        saliency[:,:,c] = single_channel_saliency(im_lab[:,:,c], hist)
    
    saliency_mean = saliency[:,:,0]*wl + saliency[:,:,1]*wa + saliency[:,:,2]*wb
    filtered_saliency = cv.GaussianBlur(saliency_mean, (3,3), sigma)
    return 1 - filtered_saliency