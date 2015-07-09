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
    self.kernels = DifferenceOfGaussians()
    self.correlations = Correlation(self.kernels.full_kernels)
    self.convolver = Convolution()
    self.MIN_IMG_WIDTH = 256
  
  def filter_image(self, img, force_homebrew=False):
    '''Perform convolution with calculated kernels
        img           => the image to convolve
        force_hombrew => if True: use my separated convolution code 
                         else: use SciPy sepfir2d
    '''
    num_kernels = len(self.kernels.full_kernels)
    img_width, img_height = img.shape
    convolved_img = {}

    for cell_type in range(num_kernels):
      if img_width < self.MIN_IMG_WIDTH or img_height < self.MIN_IMG_WIDTH:
        force_homebrew = True
      else:
        force_homebrew = False
        
      c = self.convolver.dog_sep_convolution(img, self.kernels[cell_type], 
                                             cell_type,
                                             originating_function="filter",
                                             force_homebrew=force_homebrew)
      convolved_img[cell_type] =  c

    return convolved_img

  
  def adjust_with_correlation(self, img, correlation, max_idx, max_val, 
                              is_max_val_layer=True):
      
    img_height, img_width = img.shape
    correlation_width = correlation.shape[0]
    half_correlation_width = correlation_width/2
    half_img_width = img_width/2
    half_img_height = img_height/2

    # Get max value's coordinates
    row, col = idx2coord(max_idx, img_width)
    row_idx = row/half_img_height
    col_idx = col/half_img_width
    
    # Calculate the zone to affect with the correlation
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

    img_r = [r for r in range(min_img_row, max_img_row)]
    img_c = [c for c in range(min_img_col, max_img_col)]
    
    knl_r = [r for r in range(min_knl_row, max_knl_row)]
    knl_c = [c for c in range(min_knl_col, max_knl_col)]
    
    # c_i = c_i - c_{max}<K_i, K_{max}>
    img[img_r, img_c] -= max_val*correlation[knl_r, knl_c]
        
    # mark any weird pixels as -inf so they don't matter in the search
    inf_indices = numpy.where(img[img_r, img_c] == numpy.inf)
    img[inf_indices] = 0
    img[inf_indices] -= numpy.inf
    
    nan_indices = numpy.where(img[img_r, img_c] == numpy.nan)
    img[nan_indices] = 0
    img[nan_indices] -= numpy.inf
    
    # mark max value's coordinate to -inf to get it out of the search
    if is_max_val_layer:
        img[row, col] -= numpy.inf
    
    # No need to return, values are affected because I'm doing = and -= ops


  def local_coords_to_global_idx(self, coords, cell_type, 
                                 local_img_shape, global_img_shape):
    row_add = cell_type/2
    col_add = cell_type%2
    global_coords = (coords[0] + row_add*local_img_shape[0], 
                     coords[1] + col_add*local_img_shape[1])
    global_idx = global_coords[0]*global_img_shape[1] + global_coords[1]
    return global_idx

  def global_to_single_coords(self, coords, single_shape):
    row_count = coords[0]/single_shape[0]
    col_count = coords[1]/single_shape[1]
    new_row = coords[0] - single_shape[0]*row_count
    new_col = coords[1] - single_shape[1]*col_count
    
    return (new_row, new_col)

  def cell_type_from_global_coords(self, coords, single_shape):
    row_type = coords[0]/single_shape[0]
    col_type = coords[1]/single_shape[1]
    cell_type = row_type*2 + col_type
    
    return cell_type

  def focal(self, spike_images, spikes_per_unit=0.3):
    '''Filter Overlap Correction ALgorithm, simulates the foveal pit
        region of the human retina.
        Created by Basabdatta Sen Bhattacharya.
        See DOI: 10.1109/TNN.2010.2048339
        
        spike_images => A list of the values generated by the convolution
                        procedure, stored as four 2D arrays, each with 
                        the same size/shape of the original image.
        spikes_per_unit => Percentage of the total spikes to be processed,
                           specified in a per unit [0, 1] range.
                           
        returns: an ordered list of 
                  [spike index, sorting value, cell layer/type] tuples.
    '''
    ordered_spikes = []
    
    img_size  = spike_images[0].size
    img_shape = spike_images[0].shape
    height, width = img_shape
    num_images = len(spike_images)

    #how many non-zero spikes are in the images
    max_cycles = 0
    for i in range(num_images):
        max_cycles += numpy.sum(img[i] != 0)

    total_spikes = max_cycles.copy()
    
    #reduce to desired number
    max_cycles = numpy.int(spikes_per_unit*max_cycles)
    
    #copy images from list to a large image to make it a single search space
    big_shape = (height*2, width*2)
    big_image = numpy.zeros(big_shape)
    big_coords = [(0, 0), (0, width), (height, 0), (height, width)]
    for cell_type in range(num_images):
        row, col = big_coords[cell_type]
        tmp_img = original_img[cell_type].copy()
        tmp_img[numpy.where(tmp_img == 0)] = -numpy.inf
        big_image[row:row+height, col:col+width] = tmp_img
    
    # Main FoCal loop
    for count in xrange(max_cycles):
        
        # print out completion percentage
        percent = (count*100.)/float(total_spikes-1)
        sys.stdout.write("\rFocal %d%%"%(percent))
        
        # Get maximum value's index
        max_idx = numpy.argmax(img_copies)
        # and its coordinates
        max_coords = numpy.unravel_index(max_idx, big_shape)
        # and the value
        max_val = big_image[max_coords]
        
        if max_val == numpy.inf or max_val == -numpy.inf or max_val == numpy.nan:
          sys.stderr.write("\nWrong max value in FoCal!")
          break
        
        # translate coordinates from the big image's to a single image coordinates
        if max_coords[0] < height and max_coords[1] < width:
            single_coords = max_coords
        else:
            single_coords = self.global_to_single_coords(max_coords, img_shape)
        
        # calculate a local index, to store per ganglion cell layer info 
        local_idx = single_coords[0]*width + single_coords[1]
        # calculate the type of cell from the index
        cell_type = self.cell_type_from_global_coords(max_coords, img_shape)
        
        # append max spike info to return list
        ordered_spikes.append([local_idx, max_val, cell_type])
        
        # correct surrounding pixels for overlapping kernel influence
        for overlap_cell_type in range(len(original_img)):
            # get equivalent coordinates for each layer
            overlap_idx = self.local_coords_to_global_idx(single_coords, 
                                                         overlap_cell_type, 
                                                         img_shape, big_shape)
            
            is_max_val_layer = overlap_cell_type == cell_type 
            # c_i = c_i - c_{max}<K_i, K_{max}>
            self.adjust_with_correlation_g(big_image,
                                           self.correlations[cell_type]\
                                                            [overlap_cell_type], 
                                           overlap_idx, max_val, 
                                           is_max_val_layer=is_max_val_layer)

    
    return ordered_spikes

def most_important_pixels_g(self, images, per_unit, num_kernels):

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
