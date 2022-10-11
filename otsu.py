#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 15:22:20 2020

@author: mroux
"""


import matplotlib.pyplot as plt
from skimage import data
from skimage import io as skio
from skimage.filters import threshold_otsu
import numpy as np

def histogram(im):
    print (im.shape)
    nl,nc=im.shape

    hist=np.zeros(256)

    for i in range(nl):
        for j in range(nc):
            hist[im[i][j]]=hist[im[i][j]]+1

    for i in range(256):
        hist[i]=hist[i]/(nc*nl)

    return(hist)


def otsu_thresh(im):

    h=histogram(im)

    m=0
    for i in range(256):
        m=m+i*h[i]

    maxt=0
    maxk=0


    for t in range(256):
        w0=0
        w1=0
        m0=0
        m1=0
        for i in range(t):
            w0=w0+h[i]
            m0=m0+i*h[i]
        if w0 > 0:
            m0=m0/w0

        for i in range(t,256):
            w1=w1+h[i]
            m1=m1+i*h[i]
        if w1 > 0:
            m1=m1/w1

        k=w0*w1*(m0-m1)*(m0-m1)

        if k > maxk:
            maxk=k
            maxt=t


    thresh=maxt
    binary = im > thresh
    return thresh, binary

