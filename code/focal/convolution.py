class Convolution():
  
  def sep_convolution(self, img, horz_k, vert_k, col_keep=1, row_keep=1, mode="full"):
    ''' Separated convolution -
        img      => image to convolve
        horiz_k  => first convolution kernel
        vert_k   => second convolution kernel
        col_keep => which columns are we supposed to calculate (filter with modulo)
        row_keep => which rows are we supposed to calculate
        mode     => if "full": convolve all the image otherwise just valid pixels
    '''
    half_k_width = horz_k.size/2
    half_img_width  = img.shape[1]/2
    half_img_height = img.shape[0]/2

    tmp = numpy.zeros_like(img, dtype=numpy.float32)

    if mode == "full":
      horizontal_range = xrange(img.shape[1]) 
      vertical_range   = xrange(img.shape[0])
    else:
      horizontal_range = xrange(half_k_width, img.shape[1] - half_k_width + 1)
      vertical_range   = xrange(half_k_width, img.shape[0] - half_k_width + 1)

    for y in xrange(img.shape[0]):
        for x in first_horiz_range:
            if (x - half_img_width)%col_keep != 0:
                continue

            k_sum = 0.
            k = 0

            for i in xrange(-half_k_width, half_k_width + 1):
                img_idx = x + i
                if img_idx >= 0 and img_idx < img.shape[1]:
                    k_sum += img[y,img_idx]*horz_k[k]
                k += 1

            tmp[y,x] = k_sum

    tmp2 = numpy.zeros_like(img, dtype=numpy.float32)
    for y in vertical_range:
      if (y - half_img_height)%row_keep != 0:
        continue

      for x in horizontal_range:
        if (x - half_img_width)%col_keep != 0:
          continue

        k_sum = 0.
        k = 0
        for i in xrange(-half_k_width, half_k_width + 1):
          img_idx = y + i
          if img_idx >= 0 and img_idx < img.shape[0]:
            k_sum += tmp[img_idx, x]*vert_k[k]
              
          k += 1

        tmp2[y,x] = k_sum

    return tmp2

  def dog_sep_convolution(self, img, k, cell_type, originating_function="filter",
                          sampling_resolution="basab",
                          force_homebrew = False):
    ''' Wrapper for separated convolution for DoG kernels in FoCal, 
        enables use of NumPy based sepfir2d.
        
        img       => the image to convolve
        k         => 1D kernels to use
        cell_type => ganglion cell type, useful for sampling resolution numbers
        originating_function => if "filter": use special sampling resolution,
                                otherwise use every pixel
        
    '''

    if originating_function == "filter":
        row_keep, col_keep = get_subsample_keepers(img, cell_type, k, 
                                                   sampling_resolution=sampling_resolution)
    else:
        row_keep, col_keep = 1, 1

    if not force_homebrew:
        right_img = sepfir2d(img.copy(), k[0], k[1])
        left_img  = sepfir2d(img.copy(), k[2], k[3])
    else:

        right_img = sep_convolution(img, k[0], k[1], 
                                    col_keep=col_keep, row_keep=row_keep)
        left_img  = sep_convolution(img, k[2], k[3], 
                                    col_keep=col_keep, row_keep=row_keep)

    c = left_img + right_img

    if not force_homebrew and originating_function == "filter":
        c = self, subsample(c, cell_type, k, sampling_resolution=sampling_resolution)

    return c




  def get_subsample_keepers(c, cell_type, kernel, sampling_resolution="basab"):
    if sampling_resolution == "basab":

        if cell_type > 1:
            #~ col_keep = 7
            #~ row_keep = 7
            col_keep = 5
            row_keep = 3

        else:
            col_keep = 1
            row_keep = 1

    elif sampling_resolution == "sqrt":
        col_keep = numpy.int(numpy.sqrt(kernel[0].shape[0]))
        row_keep = col_keep #numpy.int(numpy.sqrt(kernels[cell_type][0].shape[0]/2))
    elif sampling_resolution == "half":
        col_keep = kernel[0].shape[0]/2
        row_keep = col_keep
    elif sampling_resolution == "centre":
        centre_width = [3, 5, 33, 53]
        col_keep = centre_width[cell_type]/2
        row_keep = col_keep
    elif sampling_resolution == "vanrullen":
        #if cell_type > 3:
            col_keep = c.shape[1]/(2**(8 - cell_type))
            row_keep = c.shape[0]/(2**(8 - cell_type))
        #else:
            #row_keep = col_keep = 2**(cell_type+1) - 1
        #    row_keep = int((float(c.shape[0])/float(c.shape[1]))*col_keep)

    if col_keep == 0:
        col_keep = 1
    if row_keep == 0:
        row_keep = 1

    return row_keep, col_keep


  def subsample(c, cell_type, kernel, sampling_resolution="basab"):
    row_keep, col_keep = get_subsample_keepers(c, cell_type, kernel, sampling_resolution=sampling_resolution)
    
    
    #print("sub sample for cell type %s"%cell_type)
    #print("col and row keep %s, %s"%(col_keep, row_keep))
    if col_keep < c.shape[1] and row_keep < c.shape[0]:
        half_img_width = c.shape[1]/2
        half_img_height = c.shape[0]/2
        col_range = numpy.arange(c.shape[1])
        row_range = numpy.arange(c.shape[0])
#        c[:, [x for x in col_range if (x - half_img_width)%(col_keep)!= 0]] = 0
#        c[[x for x in row_range if (x - half_img_height)%(row_keep)!= 0], :] = 0
        c[:, [x for x in col_range if (x)%(col_keep)!= 0]] = 0
        c[[x for x in row_range if (x)%(row_keep)!= 0], :] = 0
    else:
        c[:,:] = 0

    #print("after subsampling procedure, before return")
    return c
