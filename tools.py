from __future__ import division
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def crop_and_resize_image(img, new_size, grayscale = False):
    height, width, channels = img.shape
    fx = new_size[0] / width
    fy = new_size[1] / height
    fxy = np.max([fx, fy])
    # half of cropsize
    csize2    = (int(np.floor(new_size[1] / fxy / 2)), int(np.floor(new_size[0] / fxy / 2)))
    center    = (width // 2, height // 2)
    img_cropped = img[(center[1] - csize2[1]):(center[1] + csize2[1]),
                      (center[0] - csize2[0]):(center[0] + csize2[0]), :]
    img_resized = cv2.resize(img,(new_size[1], new_size[0]), interpolation = cv2.INTER_CUBIC)
    if grayscale:
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR) # keep all channels
    return img_resized

def show_image(image):
    plt.axis("off")
    if len(image.shape) > 2:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(image, cmap = cm.Greys_r)
    plt.show()

def combine_images(img_in, img_out, radius):
    assert img_in.shape == img_out.shape
    h = img_in.shape[0]
    w = img_in.shape[1]
    x = np.arange(0, w); y = np.arange(0, h)
    xx, yy = np.meshgrid(x, y)
    mask_in = np.sqrt((xx - w // 2) ** 2 + (yy - h // 2) ** 2) <= radius
    img_new = np.zeros(img_in.shape, dtype = np.uint8)
    img_new[mask_in] = img_in[mask_in]
    img_new[np.logical_not(mask_in)] = img_out[np.logical_not(mask_in)]
    return img_new

def add_ring(img, radius, width, colour = (128,128,128), blur = 0):
    h = img.shape[0]
    w = img.shape[1]
    x = np.arange(0, w); y = np.arange(0, h)
    imr = np.zeros((h, w, 4), dtype = np.uint8)
    ima = np.zeros((h, w, 1), dtype = np.uint8)
    col = (colour[0], colour[1], colour[2], 255)
    cv2.circle(imr, (w // 2, h // 2), radius, col, width)    
    cv2.circle(ima, (w // 2, h // 2), radius, 255, width)    
    if blur > 0:
        #ima = cv2.blur(ima, (blur, blur))
        ima = cv2.GaussianBlur(ima, ksize = (0, 0), sigmaX = blur)
        imr[:,:,:3] = 128 
        imr[:,:,3]  = ima / (ima.max() / 255.)
    return imr
