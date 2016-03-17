from __future__ import division
import numpy as np
from skimage import io, transform, filters, color
from skimage import img_as_uint, img_as_float

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
    img_resized = transform.resize(img_cropped,(new_size[1], new_size[0]))
    if grayscale:
        img_resized = color.gray2rgb(color.rgb2gray(img_resized)) # keep 3 channels
    return img_resized

def combine_images(img_in, img_out, radius):
    assert img_in.shape == img_out.shape
    assert img_in.dtype == img_out.dtype
    h = img_in.shape[0]
    w = img_in.shape[1]
    x = np.arange(0, w); y = np.arange(0, h)
    xx, yy = np.meshgrid(x, y)
    mask_in = np.sqrt((xx - w // 2) ** 2 + (yy - h // 2) ** 2) <= radius
    img_new = np.zeros(img_in.shape, dtype = img_in.dtype)
    img_new[mask_in] = img_in[mask_in]
    img_new[np.logical_not(mask_in)] = img_out[np.logical_not(mask_in)]
    return img_new

def make_ring_for_image(img, radius, width50, colour = (128, 128, 128)):
    h = img.shape[0]
    w = img.shape[1]
    x = np.arange(0, w); y = np.arange(0, h)
    xx, yy = np.meshgrid(x, y)
    distance = np.sqrt((xx - w // 2) ** 2 + (yy - h // 2) ** 2)
    im = - (np.abs(distance - radius) - width50) / width50
    im[im < -1.] = -1   # now -1 .. +1
    im = (1. - im) * np.pi / 2.
    im = (np.cos(im) + 1.) / 2
    im4 = np.zeros((h, w, 4), dtype = img.dtype)
    col = np.array(colour).reshape((1, 1, 3)) / 256.
    im4[:, :, :3] = col
    im4[:, :, 3] = im
    return (im, im4)
    
def add_ring(img, img4):
    alpha = img4[:,:,3][..., None]
    im = img[:,:,:3] * (1 - alpha) + img4[:,:,:3] * alpha
    return im

