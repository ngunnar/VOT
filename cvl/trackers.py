import numpy as np
import cv2
import copy
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
from .image_io import crop_patch
from .dataset import BoundingBox


class NCCTracker:

    def __init__(self, learning_rate=0.1):
        self.template = None
        self.last_response = None
        self.region = None
        self.region_shape = None
        self.region_center = None
        self.learning_rate = learning_rate

    def crop_patch(self, image):
        region = self.region
        return crop_patch(image, region)

    def start(self, image, region):
        assert len(image.shape) == 2, "NCC is only defined for grayscale images"
        self.region = region
        self.region_shape = (region.height, region.width)
        self.region_center = (region.height // 2, region.width // 2)
        patch = self.crop_patch(image)

        patch = patch/255
        patch = patch - np.mean(patch)
        patch = patch / np.std(patch)

        self.template = fft2(patch)

    def detect(self, image):
        assert len(image.shape) == 2, "NCC is only defined for grayscale images"
        patch = self.crop_patch(image)
        patch = patch / 255
        patch = patch - np.mean(patch)
        patch = patch / np.std(patch)
        patchf = fft2(patch)

        responsef = np.conj(self.template) * patchf
        response = ifft2(responsef)

        r, c = np.unravel_index(np.argmax(response), response.shape)

        # Keep for visualisation
        self.last_response = response

        r_offset = np.mod(r + self.region_center[0], self.region.height) - self.region_center[0]
        c_offset = np.mod(c + self.region_center[1], self.region.width) - self.region_center[1]

        self.region.xpos += c_offset
        self.region.ypos += r_offset

        return self.region

    def update(self, image, lr=0.1):
        assert len(image.shape) == 2, "NCC is only defined for grayscale images"
        patch = self.crop_patch(image)
        patch = patch / 255
        patch = patch - np.mean(patch)
        patch = patch / np.std(patch)
        patchf = fft2(patch)
        self.template = self.template * (1 - lr) + patchf * lr


class MOSSETracker():
    def __init__(self, lr=0.125, eps=1e-6):
        self.region = None
        self.region_shape = None
        self.region_center = None
        self.last_response = None
        self.lr = lr       # recommended learning rate in  et al, 2010
        self.eps = eps     # regularization term
        self.psr = 0       # Peak-to-Sidelobe Ratio
        self.G = None
        self.A = None
        self.B = None
        self.H = None

    def start(self, image: np.ndarray, region: BoundingBox):
        self.region = region
        self.region_shape = (region.height, region.width)
        self.region_center = (region.height // 2, region.width // 2)

        patch = self._crop_patch(image)

        g = np.zeros((region.height, region.width), np.float32)
        g[region.height // 2, region.width // 2] = 1
        g = cv2.GaussianBlur(g, (-1, -1), 2.0)
        g /= g.max()
        self.G = fft2(g)

        self.A = np.zeros_like(self.G)
        self.B = np.zeros_like(self.G)

        n = 8    # number of times to repeat training on 1st frame
        for i in range(n):
            f = self._pre_process(self._perturb_img(patch))
            F = fft2(f)
            self.A += self.G * np.conjugate(F)
            self.B += F * np.conjugate(F)

        self._update_filter()

    def detect(self, image):
        patch = self._crop_patch(image)
        f = self._pre_process(patch)
        self.last_response, (dw, dh), self.psr = self._correlate(f)

        if self.psr < 8:
            return self.region

        self.region_center = (self.region_center[0] + dh, self.region_center[1] + dw)
        self.region.xpos += dw
        self.region.ypos += dh

        return self.region

    def update(self, image, lr=0.125):
        self.lr = lr
        patch = self._crop_patch(image)
        f = self._pre_process(patch)
        F = fft2(f)

        self.A = self.lr * self.G * np.conjugate(F) + (1 - self.lr) * self.A
        self.B = self.lr * F * np.conjugate(F) + (1 - self.lr) * self.B
        self._update_filter()

    def _crop_patch(self, image: np.ndarray):
        return crop_patch(image, self.region)

    def _pre_process(self, image: np.ndarray):
        image = np.log(np.float32(image) + 0.1)  # adding a small number for stability
        image = (image - image.mean()) / image.std() + 1e-5
        window = cv2.createHanningWindow((image.shape[1], image.shape[0]), cv2.CV_32F)
        # TODO: change this later for multichannel
        image = image * window
        return image

    def _perturb_img(self, image):
        h, w = image.shape[:2]
        # construct the tranformation matrix
        transform_matrix = np.zeros((2, 3))

        # construct the rotation part
        k = 0.1
        theta = np.random.uniform(-k, k)
        c = np.cos(theta)
        s = np.sin(theta)
        transform_matrix[:2, :2] = [[c, -s], [s, c]]
        transform_matrix[:2, :2] += (np.random.rand(2, 2) - 0.5) * k  # pertubation

        # Add the translation part
        transform_matrix[:, -1] += np.random.uniform(-k, k, 2)

        return cv2.warpAffine(image, transform_matrix, (w, h), borderMode=cv2.BORDER_REFLECT)

    def _update_filter(self):
        self.H = np.conjugate(self.A / (self.B + self.eps))

    def _correlate(self, f):
        F = fft2(f)
        G = F * np.conjugate(self.H)
        g = ifft2(G)
        h, w = g.shape
        max_loc = np.unravel_index(g.argmax(), g.shape)
        g_max = g[max_loc]
        g_rest = copy.deepcopy(g)
        # cut out the area containing the peak
        g_rest[max_loc[0]-5:max_loc[0]+5, max_loc[1]-5:max_loc[1]+5] = 0
        psr = (g_max - g_rest.mean()) / g_rest.std()

        # TODO: check if the order of coordinates is correct
        return g, (max_loc[1] - w//2, max_loc[0] - h//2), psr
