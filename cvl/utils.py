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
    imgP = [np.pad(img[:,:,i], [padY, padX], 'constant', constant_values=(np.median(img[:,:,i]))) for i in range(img.shape[-1])]
    imgR = [rotate(img, angle, reshape=False, mode='constant') for img in imgP]
    imgR = np.asarray(imgR)
    imgR = np.moveaxis(imgR, 0, -1)
    return imgR[padY[0]: -padY[1], padX[0]: -padX[1]]


def plot(image, response, region, name):
    import matplotlib.pyplot as plt
    def transparent_cmap(cmap, N=255):
        "Copy colormap and set alpha values"

        mycmap = cmap
        mycmap._init()
        mycmap._lut[:,-1] = np.linspace(0, 0.8, N+4)
        return mycmap


    #Use base cmap to create transparent
    mycmap = transparent_cmap(plt.cm.Reds)

    h = image.shape[0]
    w = image.shape[1]
    y, x = np.mgrid[0:h, 0:w]

    r = np.zeros((image.shape[0], image.shape[1]))
    xmin = region.xpos
    ymin = region.ypos
    xmax = xmin + region.width
    ymax = ymin + region.height
    if xmin < 0:
        dx = -xmin
        xmin = 0
    else:
        xmin = xmin
        dx = 0
    
    if ymin < 0:
        dy = -ymin
        ymin = 0
    else:
        ymin = ymin
        dy = 0
    
    if ymax >= image.shape[0]:
        ymax = image.shape[0]
        dyy = ymin - ymax
    else:
        ymax = ymax
        dyy = response.shape[0]
    
    if xmax >= image.shape[1]:
        xmax = image.shape[1]
        dxx = xmax - xmin
    else:
        xmax = xmax
        dxx = response.shape[1]

    try:
        r[ymin:ymax, xmin:xmax] = response[dy:dyy, dx:dxx]
    except:
        a = 1
    
    fig, ax = plt.subplots(1, 1)

    ax.imshow(image)
    cb = ax.contourf(x,y,r, 15, cmap=mycmap)
    ax.axis('off')
    plt.savefig('./images/{0}.png'.format(name))
    plt.close()