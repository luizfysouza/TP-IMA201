from numba import jit 
import numpy as np
import skimage as ski
from skimage.segmentation import slic

@jit(nopython=True)
def normalize(im):
    return (im - im.min()) / (im.max() - im.min())

@jit(nopython=True)
def pixel_entropy(im, prob):
    entropy = np.zeros(im.shape, dtype=float)
    for i in range(entropy.shape[0]):
        for j in range(entropy.shape[1]):
            if prob[im[i,j]] == 0:
                entropy[i,j] = 0
            else:
                entropy[i,j] = - prob[im[i,j]] * np.log2(prob[im[i,j]])
    entropy = normalize(entropy)
    return entropy

#@jit(nopython=True)
def superpixel_entropy(entropy, segments):
#    entropy_values = (segments.max()+1) * [[]]
#    for i in range(segments.shape[0]):
#        for j in range(segments.shape[1]):
#            entropy_values[segments[i,j]].append(entropy[i,j])
    #print("entropy values", entropy_values[0], entropy_values[1])

    superpixel_entropy = np.zeros(segments.max()+1, dtype=float)
    for i in range(segments.max()+1):
        mask = segments == i
        masked_entropy = mask * entropy
        superpixel_entropy[i] = np.sum(masked_entropy)
        #superpixel_entropy[i] = sum(entropy_values[i]) #/ len(entropy_values[i])
        
    #print("superpixel entropy", superpixel_entropy)

    superpixel_entropy_image = np.zeros(segments.shape, dtype=float)
    for i in range(segments.shape[0]):
        for j in range(segments.shape[1]):
            superpixel_entropy_image[i,j] = superpixel_entropy[segments[i,j]]

    superpixel_entropy_image = normalize(superpixel_entropy_image)
    return superpixel_entropy_image

def information_saliency(im, n_segments = 1000, sigma = 1, wl = 1./3, wa = 1./3, wb = 1./3):
    im_lab = ski.color.rgb2lab(im)
    segments = slic(im_lab, n_segments = n_segments, sigma = sigma, convert2lab=False)
    #show(mark_boundaries(im, segments))

    superpixel_entropy_image = [None] * im.shape[2]
    for c in range(im.shape[2]):  
        hist, bin_edges = np.histogram(im[:,:,c], bins=256)
        prob = hist / im.size
        pixel_entropy_image = pixel_entropy(im[:,:,c], prob)
        superpixel_entropy_image[c] = superpixel_entropy(pixel_entropy_image, segments)
        
        #show(pixel_entropy_image)
        #show(superpixel_entropy_image[c])

    entropy = superpixel_entropy_image[0]*wl + superpixel_entropy_image[1]*wa + superpixel_entropy_image[2]*wb
    entropy = np.array(entropy)
    return entropy