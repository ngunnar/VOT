import numpy as np
import cv2

from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
from scipy.stats import multivariate_normal
from scipy.ndimage import rotate
from .image_io import crop_patch
from .utils import pre_process, rotateImage, plot, get_search_region

########################################################################################################################
# Grayscale MOSSE
########################################################################################################################


class GrayscaleMosseTracker:
    """
    implementation according to paper: Bolme et al., Visual Object Tracking using Adaptive Correlation Filters, 2010
    https://www.cs.colostate.edu/~vision/publications/bolme_cvpr10.pdf
    """
    def __init__(self, 
                 learning_rate=0.125, 
                 epsilon=1e-3,
                 search_size = 1,
                 sigma = 2.0,
                 save_img=False,
                 name='grayscale',
                save_frame = 10):
        self.patch_fft2 = None
        self.last_response = None
        self.region = None
        self.region_shape = None
        self.region_center = None

        self.lr = learning_rate  # for incremental MOSSE
        self.epsilon = epsilon
        self.sigma = sigma / search_size

        self.search_size = search_size
        self.save_img = save_img
        self.name = name
        self.save_frame = save_frame

    def crop_patch(self, image):
        region = self.search_region
        return crop_patch(image, region)

    def start(self, image_color, region):
        """
        image: numpy array of shape (width, height) for complete image
        region: bounding box of shape (width, height)

        Explanations:
        F as 2D Fourier transform of input image f
        H as 2D Fourier transform of filter h
        G = F * H^{*}, where {*} denotes the complex conjugate
        """
        image = np.sum(image_color, 2) / 3
        assert len(image.shape) == 2, "Grayscale MOSSE is only defined for grayscale images"
        self.frame = 0
        self.region = region
        self.region_shape = (region.height, region.width)
        self.region_center = (region.height // 2, region.width // 2)

        self.search_region = region.rescale(self.search_size,round_coordinates=True)
        self.search_region_shape = (self.search_region.height, self.search_region.width)
        self.search_region_center = (self.search_region.height // 2, self.search_region.width // 2)

        patch = self.crop_patch(image)

        # pre-process the patch
        patch = pre_process(patch)
        # Fourier transform of patch
        F = fft2(patch)

        # initialise G as 2D Gaussian centered on the target
        Sigma = np.eye(2) * self.sigma ** 2
        mu = [self.search_region_center[0], self.search_region_center[1]]
        x, y = np.mgrid[0:self.search_region.height:1, 0:self.search_region.width:1]
        pos = np.dstack((x, y))
        r = multivariate_normal(mu, Sigma)
        g = r.pdf(pos)
        self.G = fft2(g)

        # compute numerator A and denominator B for H = A/B
        A = self.G * np.conj(F)
        B = F * np.conj(F)
        
        image_center = (self.region.xpos + self.region_center[1], self.region.ypos + self.region_center[0])
        for angle in np.arange(-20,20,5):
            img_tmp = rotateImage(image_color, angle, image_center) # Rotate
            img_tmp = np.sum(img_tmp, 2) / 3
            patch = self.crop_patch(img_tmp)
            patch = pre_process(patch)
            F = fft2(patch)
            A += self.G * np.conj(F)
            B += F * np.conj(F)                
        
        # compute filter
        self.A = A
        self.B = B
        self.H_conj = self.A / (self.B + self.epsilon)

        if self.save_img and self.frame % self.save_frame == 0:
            plot(image_color, g.real, self.search_region, "{0}_{1}".format(self.name, self.frame))

    def detect(self, image_color):
        """
        image: numpy array of shape (width, height) for complete image
        """
        image = np.sum(image_color, 2) / 3
        assert len(image.shape) == 2, "Grayscale MOSSE is only defined for grayscale images"
        self.frame += 1
        patch = self.crop_patch(image)
        # pre-process the patch
        patch = pre_process(patch)
        # Fourier transform of patch
        F = fft2(patch)

        # get response
        responsef = F * self.H_conj
        response = ifft2(responsef)

        if self.save_img and self.frame % self.save_frame == 0:
            plot(image_color, response, self.search_region, "{0}_{1}".format(self.name, self.frame))
        
        # get indices of max value
        r, c = np.unravel_index(np.argmax(response), response.shape)

        # update shift in center of the region
        r_offset = r - self.region_center[0]
        c_offset = c - self.region_center[1]
        self.region.xpos += c_offset
        self.region.ypos += r_offset

        self.search_region.xpos += c_offset
        self.search_region.ypos += r_offset

        return self.region

    def update(self, image_color):
        """
        output has to be an updated region.xpos, region.ypos, region.width, region.height.
        xpos, ypos is the pixel on the lower left end of the bounding box
        """
        image = np.sum(image_color, 2) / 3
        assert len(image.shape) == 2, "Grayscale MOSSE is only defined for grayscale images"
        patch = self.crop_patch(image)
        # pre-process the patch
        patch = pre_process(patch)
        # Fourier transform of patch
        F = fft2(patch)

        # compute factors for H
        self.A = self.lr * self.G * np.conj(F) + (1-self.lr) * self.A
        self.B = self.lr * F * np.conj(F) + (1-self.lr) * self.B

        #self.A = np.conj(self.G) * F + self.A
        #self.B = F * np.conj(F) + self.B

        self.H_conj = self.A / (self.B + self.epsilon)
