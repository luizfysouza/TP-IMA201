from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt
import argparse
import otsu
import numpy as np
from skimage.filters import threshold_otsu
from skimage.color import rgb2lab

#Load arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
ap.add_argument("-s", "--segments", required = True, type = int, help = "Number of segments")
args = vars(ap.parse_args())
im = img_as_float(io.imread(args["image"]))

#Convert to LAB
im_lab = rgb2lab(im/255)


#Make super pixels
segments = slic(im, n_segments = args["segments"], sigma = 5, compactness = 10, convert2lab = True)
plt.imshow(mark_boundaries(im, segments))
plt.show()


segments = slic(im, n_segments = args["segments"], sigma = 5, compactness = 10, convert2lab = False)
plt.imshow(mark_boundaries(im, segments))
plt.show()

#t, bin = otsu.otsu_thresh(im[:,:,0])
#plt.imshow(bin)
t = threshold_otsu(im_lab[:,:,0])
bin = im_lab[:,:,0] > t
plt.imshow(bin)
plt.show()
