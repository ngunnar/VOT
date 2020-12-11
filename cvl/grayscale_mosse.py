import numpy as np
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
from scipy.stats import multivariate_normal
from scipy.ndimage import rotate
from .image_io import crop_patch
import matplotlib.pyplot as plt
from .utils import pre_process

########################################################################################################################
# Grayscale MOSSE
########################################################################################################################


class GrayscaleMosseTracker:
    """
    implementation according to paper: Bolme et al., Visual Object Tracking using Adaptive Correlation Filters, 2010
    https://www.cs.colostate.edu/~vision/publications/bolme_cvpr10.pdf
    """
    def __init__(self, learning_rate=0.125, epsilon=1e-3):
        self.patch_fft2 = None
        self.last_response = None
        self.region = None
        self.region_shape = None
        self.region_center = None

        self.lr = learning_rate  # for incremental MOSSE
        self.epsilon = epsilon

    def crop_patch(self, image):
        region = self.region
        return crop_patch(image, region)

    def start(self, image, region):
        """
        image: numpy array of shape (width, height) for complete image
        region: bounding box of shape (width, height)

        Explanations:
        F as 2D Fourier transform of input image f
        H as 2D Fourier transform of filter h
        G = F * H^{*}, where {*} denotes the complex conjugate
        """
        assert len(image.shape) == 2, "Grayscale MOSSE is only defined for grayscale images"
        self.region = region
        self.region_shape = (region.height, region.width)
        self.region_center = (region.height // 2, region.width // 2)
        patch = self.crop_patch(image)

        # pre-process the patch
        patch = pre_process(patch)
        # Fourier transform of patch
        F = fft2(patch)

        # initialise G as 2D Gaussian centered on the target
        sigma_g = 2.0
        Sigma = np.eye(2) * sigma_g ** 2
        mu = [self.region_center[0], self.region_center[1]]
        x, y = np.mgrid[0:region.height:1, 0:region.width:1]
        pos = np.dstack((x, y))
        r = multivariate_normal(mu, Sigma)
        g = r.pdf(pos)
        self.G = fft2(g)

        # compute numerator A and denominator B for H = A/B
        self.A = self.G * np.conj(F)
        self.B = F * np.conj(F)

        # compute filter
        self.H_conj = self.A / (self.B + self.epsilon)

    def detect(self, image):
        """
        image: numpy array of shape (width, height) for complete image
        """
        assert len(image.shape) == 2, "Grayscale MOSSE is only defined for grayscale images"
        patch = self.crop_patch(image)
        # pre-process the patch
        patch = pre_process(patch)
        # Fourier transform of patch
        F = fft2(patch)

        # get response
        responsef = F * self.H_conj
        response = ifft2(responsef)

        # get indices of max value
        r, c = np.unravel_index(np.argmax(response), response.shape)

        # update shift in center of the region
        r_offset = r - self.region_center[0]
        c_offset = c - self.region_center[1]
        self.region.xpos += c_offset
        self.region.ypos += r_offset

        return self.region

    def update(self, image):
        """
        output has to be an updated region.xpos, region.ypos, region.width, region.height.
        xpos, ypos is the pixel on the lower left end of the bounding box
        """
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
