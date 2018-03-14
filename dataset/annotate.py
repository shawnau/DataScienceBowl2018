from common import *
import numpy as np


def multi_mask_to_color_overlay(multi_mask, image=None, color=None):

    height,width = multi_mask.shape[:2]
    overlay = np.zeros((height,width,3),np.uint8) if image is None else image.copy()
    num_masks = int(multi_mask.max())
    if num_masks==0: return overlay

    if type(color) in [str] or color is None:
        #https://matplotlib.org/xkcd/examples/color/colormaps_reference.html

        if color is None: color='summer'  #'cool' #'brg'
        color = plt.get_cmap(color)(np.arange(0,1,1/num_masks))
        color = np.array(color[:,:3])*255
        color = np.fliplr(color)
        #np.random.shuffle(color)

    elif type(color) in [list,tuple]:
        color = [ color for i in range(num_masks) ]

    for i in range(num_masks):
        mask = multi_mask==i+1
        overlay[mask]=color[i]
        #overlay = instance[:,:,np.newaxis]*np.array( color[i] ) +  (1-instance[:,:,np.newaxis])*overlay

    return overlay


def multi_mask_to_contour_overlay(multi_mask, image=None, color=[255,255,255]):

    height,width = multi_mask.shape[:2]
    overlay = np.zeros((height,width,3),np.uint8) if image is None else image.copy()
    num_masks = int(multi_mask.max())
    if num_masks==0: return overlay

    for i in range(num_masks):
        mask = multi_mask==i+1
        contour = mask_to_inner_contour(mask)
        overlay[contour]=color

    return overlay


def mask_to_outer_contour(mask):
    pad = np.lib.pad(mask, ((1, 1), (1, 1)), 'reflect')
    contour = (~mask) & (
            (pad[1:-1,1:-1] != pad[:-2,1:-1]) \
          | (pad[1:-1,1:-1] != pad[2:,1:-1])  \
          | (pad[1:-1,1:-1] != pad[1:-1,:-2]) \
          | (pad[1:-1,1:-1] != pad[1:-1,2:])
    )
    return contour


def mask_to_inner_contour(mask):
    pad = np.lib.pad(mask, ((1, 1), (1, 1)), 'reflect')
    contour = mask & (
            (pad[1:-1,1:-1] != pad[:-2,1:-1]) \
          | (pad[1:-1,1:-1] != pad[2:,1:-1])  \
          | (pad[1:-1,1:-1] != pad[1:-1,:-2]) \
          | (pad[1:-1,1:-1] != pad[1:-1,2:])
    )
    return contour
