import numpy as np
from scipy.ndimage import rotate

def window_func_2d(height, width):
    win_col = np.hanning(width)
    win_row = np.hanning(height)
    mask_col, mask_row = np.meshgrid(win_col, win_row)
    win = mask_col * mask_row
    return win


def pre_process(img):
    img = img / 255
    # Get size of image
    h, w = img.shape
    # Transform pixel values using log function which helps with low contrast lightning
    img = np.log(img + 1)
    # The pixels are normalized to have a mean of 0.0 and norm of 1.
    img = (img - np.mean(img)) / np.std(img)
    # Finally the image is mulitplied by a consine window which gradually reduce the pixel values near the edge to zero.
    return img * window_func_2d(h, w)


def rotateImage(img, angle, pivot):
    padX = [img.shape[1] - pivot[0], pivot[0]]
    padY = [img.shape[0] - pivot[1], pivot[1]]
    imgP = np.pad(img, [padY, padX], 'constant')
    imgR = rotate(imgP, angle, reshape=False, mode='constant')
    return imgR[padY[0]: -padY[1], padX[0]: -padX[1]]