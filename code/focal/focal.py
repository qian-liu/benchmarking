# coding: utf-8

import numpy
from scipy.signal import sepfir2d, convolve2d
from scipy.misc import imresize
import matplotlib.image as mpimg
from matplotlib import cm
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import pickle
from os import listdir
from os.path import isfile, join
import sys
import md5








class Focal():
  
  def rgb2gray(self, rgb):
    #return numpy.int16(numpy.dot(rgb[:,:,:3], [0.299, 0.587, 0.144]))
    #return numpy.floor(numpy.dot(rgb[:,:,:3], [0.299, 0.587, 0.144]))
    return rgb[:,:,0]*0.299 + rgb[:,:,1]*0.587 + rgb[:,:,2]*0.144

  def idx2coord(self, idx, width):
    return (int(idx/width), idx%width)
  
  def adjust_with_correlation_g(self, img, correlation, max_idx, max_val, is_max_val_layer=True):
      
    img_height, img_width = img.shape
    correlation_width = correlation.shape[0]
    half_correlation_width = correlation_width/2
    half_img_width = img_width/2
    half_img_height = img_height/2
    
    row, col = idx2coord(max_idx, img_width)
    row_idx = row/half_img_height
    col_idx = col/half_img_width
    
    up_lim = (row_idx)*half_img_height
    left_lim = (col_idx)*half_img_width
    
    down_lim = (row_idx + 1)*half_img_height
    right_lim = (col_idx + 1)*half_img_width
    
    max_img_row = numpy.min([down_lim - 1, row + half_correlation_width + 1])
    max_img_col = numpy.min([right_lim  - 1, col + half_correlation_width + 1])
    min_img_row = numpy.max([up_lim, row - half_correlation_width])
    min_img_col = numpy.max([left_lim, col - half_correlation_width])
    
    max_img_row_diff = max_img_row - row
    max_img_col_diff = max_img_col - col
    min_img_row_diff = row - min_img_row
    min_img_col_diff = col - min_img_col
    
    min_knl_row = half_correlation_width - min_img_row_diff
    min_knl_col = half_correlation_width - min_img_col_diff
    max_knl_row = half_correlation_width + max_img_row_diff
    max_knl_col = half_correlation_width + max_img_col_diff

    
    try:
        img[min_img_row:max_img_row, min_img_col:max_img_col] -= \
        max_val*correlation[min_knl_row:max_knl_row, min_knl_col:max_knl_col]
        
    except ValueError:
        print("cell_type", cell_type)
        print("row, col indices", row_idx, col_idx)
        print("idx", max_idx)
        print("Up/Left", up_lim, left_lim)
        print("Down/Right", down_lim, right_lim)
        print("shape", img.shape)
        print("row, col", row, col)
        print("img", min_img_row, max_img_row, min_img_col, max_img_col)
        print("cor", min_knl_row, max_knl_row, min_knl_col, max_knl_col)
        print("val", max_val)
        print("val*cor", max_val*correlation[min_knl_row:max_knl_row, min_knl_col:max_knl_col])
        print("img", img[min_img_row:max_img_row, min_img_col:max_img_col])
        raise Exception("Broadcasting?")
    
    inf_indices = numpy.where(img[min_img_row:max_img_row, min_img_col:max_img_col] == numpy.inf)
    img[inf_indices] = 0
    img[inf_indices] -= numpy.inf
    
    nan_indices = numpy.where(img[min_img_row:max_img_row, min_img_col:max_img_col] == numpy.nan)
    img[nan_indices] = 0
    img[nan_indices] -= numpy.inf
    
    if is_max_val_layer:
        img[row, col] -= numpy.inf
    


