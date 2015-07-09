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

from dog import DifferenceOfGaussians
from convolution import Convolution
from correlation import Correlation

class Focal():
  
  def __init__(self):
    self.kernels = 
    
  
  def filter_image(img, kernels, num_kernels, sampling_resolution="basab",
                 force_homebrew = True, image_pyramid=False,use_mirrors=False):
    '''
    sampling_resolution -> "basab" (every pixel for midget, every 5 COLS and 3 ROWS for parasol)
                        -> "sqrt"  (every sqrt(kernel_width) COLS, every sqrt(kernel_width/2) ROWS for all cell types)
                        -> "half"  (every kernel_width/2 COLS&ROWS for all cell types)
    '''
    img_width, img_height = img.shape
    convolved_img = {}

    old_img = img.copy()
    for cell_type in range(num_kernels):
        if kernels[cell_type][0].size > MAX_FILTER_WIDTH:
            force_homebrew = True
        else:
            force_homebrew = False
        c = dog_sep_convolution(img, kernels[cell_type], cell_type,
                                originating_function="filter",
                                sampling_resolution=sampling_resolution,
                                force_homebrew = force_homebrew)
        
        if image_pyramid == True:
            old_img = c.copy()

        
        #convolved_img[cell_type] = c
        if use_mirrors:
            convolved_img[cell_type] = c*(c>0)
            convolved_img[num_kernels + cell_type] = numpy.abs(c)*(c<0)
        else:
            convolved_img[cell_type] =  c#*(c<0)

    return convolved_img

  
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
    

  def local_coords_to_global_idx(coords, cell_type, 
                               local_img_shape, global_img_shape):
    row_add = cell_type/2
    col_add = cell_type%2
    global_coords = (coords[0] + row_add*local_img_shape[0], 
                     coords[1] + col_add*local_img_shape[1])
    global_idx = global_coords[0]*global_img_shape[1] + global_coords[1]
    return global_idx

  def global_to_single_coords(coords,  single_shape):
    row_count = coords[0]/single_shape[0]
    col_count = coords[1]/single_shape[1]
    new_row = coords[0] - single_shape[0]*row_count
    new_col = coords[1] - single_shape[1]*col_count
    
    return (new_row, new_col)

  def cell_type_from_global_coords(coords, single_shape):
    row_type = coords[0]/single_shape[0]
    col_type = coords[1]/single_shape[1]
    cell_type = row_type*2 + col_type
    
    return cell_type

  def focal1_g(original_img, correlation_LUTs, num_kernels, useful_spikes_perunit=0.2, 
           use_old_val=False):
    ordered_spikes = []
    
    img_size = original_img[0].size
    img_shape = original_img[0].shape
    #max_cycles = numpy.int(numpy.round(useful_spikes_perunit*img_size*num_kernels))
    max_cycles = 0
    for i in range(len(original_img)):
        max_cycles += numpy.sum(original_img[i] != 0)

    total_spikes = 0
    total_spikes += max_cycles
    
    max_cycles = numpy.int(useful_spikes_perunit*max_cycles)
    big_shape = (img_shape[0]*2, img_shape[1]*2)
    img_copies = numpy.zeros(big_shape)
    big_coords = [(0, 0), (0, img_shape[1]), (img_shape[0], 0), (img_shape[0], img_shape[1])]
    for cell_type in range(len(original_img)):
        height, width = img_shape
        row, col = big_coords[cell_type]
        tmp_img = original_img[cell_type].copy()
        tmp_img[numpy.where(tmp_img == 0)] = -numpy.inf
        img_copies[row:row+height, col:col+width] = tmp_img
    
    for count in xrange(max_cycles):
        
        percent = (count*100.)/float(total_spikes-1)
        sys.stdout.write("\rFocal %d%%"%(percent))
        
        max_idx = numpy.argmax(img_copies)
        
        max_coords = numpy.unravel_index(max_idx, big_shape)
        if max_coords[0] < img_shape[0] and max_coords[1] < img_shape[1]:
            single_coords = max_coords
        else:
            single_coords = global_to_single_coords(max_coords, img_shape)
        
        local_idx = single_coords[0]*img_shape[1] + single_coords[1]
            
        cell_type = cell_type_from_global_coords(max_coords, img_shape)
        
        
        
        if use_old_val:
            max_val = original_img[cell_type][single_coords]
        else:
            max_val = img_copies[max_coords]

        if img_copies[max_coords] == numpy.inf:
            #~ print "max is inf"
            break
        
        if img_copies[max_coords] == -numpy.inf:
            #~ print "max is -inf"
            break
        
        if img_copies[max_coords] == numpy.nan:
            #~ print "max is NaN"
            break
            
        #print("max_idx, max_coords, single_coords, local_idx, cell_type, max_val")
        #print(max_idx, max_coords, single_coords, local_idx, cell_type, max_val)
        
        ordered_spikes.append([local_idx, max_val, cell_type])

        for overlap_cell_type in range(len(original_img)):
            #print("overlap on cell type ", overlap_cell_type)
            overlap_idx = local_coords_to_global_idx(single_coords, overlap_cell_type, 
                                                     img_shape, big_shape)
            is_max_val_layer = overlap_cell_type == cell_type 
            adjust_with_correlation_g(img_copies,
                                      correlation_LUTs[cell_type][overlap_cell_type], 
                                      overlap_idx, max_val, is_max_val_layer=is_max_val_layer)

    
    #plot_images(original_img[0], img_copies)
    
    return ordered_spikes

def most_important_pixels_g(images, per_unit, num_kernels):

    img_size = images[0].size
    
    reverse_sort = True
    for cell_type in range(len(images)):
        img = images[cell_type].copy()
        img = img.reshape(img.size)
        if cell_type == 0:
            pixel_list = numpy.asarray([[x, img[x], cell_type] \
                         for x in range(img_size) if img[x] != 0])
        else:
            new_pix_list = numpy.asarray([[x, img[x], cell_type] \
                         for x in range(img_size) if img[x] != 0])
            pixel_list = numpy.append(pixel_list, new_pix_list, axis = 0)
    
    #print pixel_list
    sorted_indices = sorted(range(len(pixel_list)), 
                            key=lambda k: pixel_list[k][1], 
                            reverse=reverse_sort) #True means max to min

    total_pixels = len(pixel_list)
    num_pixels = int(per_unit*total_pixels)
    
    #~ print("most important pixels count %s of %s"%(num_pixels, total_pixels))

    return pixel_list[sorted_indices[:num_pixels]]
