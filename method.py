from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt
import argparse
import numpy as np
from skimage.filters import threshold_otsu
from skimage.color import rgb2lab
from entropy import information_saliency
from statisticalsaliency import statistical_saliency
from entropy import normalize
from thresholds import otsu, km

#Load arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
ap.add_argument("-s", "--segments", type = int, default=1000, help = "Number of segments")
ap.add_argument("-w", "--weights", type = float, nargs=3, default=[1, 0, 0], help = "Weights for L, A and B channels")
ap.add_argument("-f", "--fusion", type=str, default="mean", help="Metric to fuse the 2 saliencies: max, min, or mean")
ap.add_argument("-t", "--threshold", type=str, default="otsu", help = "Method to find image's threshold")
ap.add_argument("-o", "--shift", type=float, default=0, help="Shift to the right if Otsu is selected.")
args = vars(ap.parse_args())
im = img_as_float(io.imread(args["image"]))


#Get saliency
saliency = normalize(statistical_saliency(im, wl = args["weights"][0], wa = args["weights"][1],
                                          wb = args["weights"][2], sigma=0))
#saliency = normalize(statistical_saliency(im, wl = 1, wa = 0,
#                                          wb = 0, sigma=0)

#Get entropy
entropy = information_saliency(im, args["segments"])

#Fusion of the images
if (args['fusion'] == "mean"):
    new_im = (entropy + saliency)/2
if (args['fusion'] == 'max'):
    new_im = max(entropy, saliency)
if (args['fusion'] == 'min'):
    new_im = min(entropy, saliency)

#Apply threshold methods
if (args['threshold'] == "otsu"):
    im_bin = otsu(new_im, args['shift'])
else:
    im_bin = km(new_im, 2)

#Show results
plt.figure()
plt.subplot(2,2,1)
plt.imshow(im)
plt.title("Original image")
plt.subplot(2,2,2)
plt.imshow(saliency)
plt.title("Saliency")
plt.subplot(2,2,3)
plt.imshow(entropy)
plt.title("Entropy")
plt.subplot(2,2,4)
plt.imshow(im_bin)
plt.title("Thresholded image")
plt.show()
