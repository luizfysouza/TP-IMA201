# TP-IMA201

Example:

python method.py --image imgs/moscou.jpg --segments 5000 --weights 1 0 0 --fusion max --threshold otsu --shift 10

Where:

--image is the path of the target image.

--segments is the number of segments used in the SLIC algorithm. (Default 10000)

--weights is the weight of the channels L,A and B used in the weighted average for the entropy method. (Default 1 0 0)

--fusion is the method used to the image's fusion after the statistical saliency and the entropy. (Default mean)

--threshold selects between otsu and k-means to find the image's threshold. (Default otsu)

--shift only works for the otsu method, it shifts the threshold found x to the right in the histogram. (Default 0)

Only the image parameter is required, all the other parameters have a default value that the authors found to yield the best results.
