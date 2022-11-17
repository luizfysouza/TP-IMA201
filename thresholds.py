from skimage.filters import threshold_otsu
import numpy as np
from sklearn.utils import shuffle
from sklearn.cluster import KMeans


#Otsu
def otsu(im, shift):
    t = threshold_otsu(im)
    return im>(t+shift)

#K-means
def recreate_image(codebook, labels, w, h):
    image = np.zeros((w, h))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    return image


def km (im, n_class):
    n_class = n_class

# Load Image and transform to a 2D numpy array.
    w, h = original_shape = tuple(im.shape)
    d = 1
    image_array = np.reshape(im, (w * h, d))
    image_array_sample = shuffle(image_array, random_state=0)[:1000]
    kmeans = KMeans(n_clusters=n_class, random_state=0).fit(image_array_sample)

    labels = kmeans.predict(image_array)
    
    return recreate_image(kmeans.cluster_centers_, labels, w, h)