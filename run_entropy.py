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
from skimage import io as skio
from PIL import Image

#Load arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
ap.add_argument("-s", "--segments", type = int, default=10000, help = "Number of segments")
ap.add_argument("-w", "--weights", type = float, nargs=3, default=[1, 0, 0], help = "Weights for L, A and B channels")
ap.add_argument("-f", "--fusion", type=str, default="mean", help="Metric to fuse the 2 saliencies: max, min, or mean")
ap.add_argument("-t", "--threshold", type=str, default="otsu", help = "Method to find image's threshold")
ap.add_argument("-o", "--shift", type=float, default=0, help="Shift to the right if Otsu is selected.")
args = vars(ap.parse_args())
#im = img_as_float(io.imread(args["image"]))
im = skio.imread(args['image'])


#Get entropy
entropy = information_saliency(im, args["segments"])

print("type entropy", type(entropy)," - " ,entropy.shape)

Image.fromarray(entropy).save(f"results/{args['image']}_{args['segments']}seg_entropy.png")


