import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
from scipy.stats import multivariate_normal
from scipy.ndimage import rotate
from .image_io import crop_patch
from .utils import pre_process, rotateImage, plot

########################################################################################################################
# RGB Mosse
########################################################################################################################

class MultiMosseTracker():
    def __init__(self, 
                 learning_rate = 0.125,
                 epsilon=1e-3,
                 search_size = 1.0,
                 sigma = 2.0,
                 save_img=False,
                 name='rgb',
                 save_frame = 10):
        self.epsilon = epsilon
        self.learning_rate = learning_rate 
        self.sigma = sigma/search_size
        self.search_size = search_size
        self.save_img = save_img
        self.name = name
        self.save_frame = save_frame

    def get_patch(self, features):
        region = self.search_region
        return [crop_patch(c, region) for c in features]

    def preprocess_data(self, features):
        return np.asarray([pre_process(c) for c in features])

    def start(self, image, region):
        self.frame = 0
        features = [image[...,i] for i in range(image.shape[-1])]
        assert len(features) == 3, print(len(features))
        # Image is the first frame
        # Region is the bounding box around target in first frame
        self.region = region
        self.region_shape = (region.height, region.width)
        self.region_center = (region.height // 2, region.width // 2)
        
        self.search_region = region.rescale(self.search_size,round_coordinates=True)
        self.search_region_shape = (self.search_region.height, self.search_region.width)
        self.search_region_center = (self.search_region.height // 2, self.search_region.width // 2)

        # Extract patches from image
        f = self.get_patch(features)
        f = self.preprocess_data(f)
        F = fft2(f)

        # Create desired response
        Sigma = np.eye(2) * self.sigma ** 2
        mu = [self.search_region_center[0], self.search_region_center[1]]
        x, y = np.mgrid[0:self.search_region.height:1,
               0:self.search_region.width:1]
        pos = np.dstack((x, y))
        r = multivariate_normal(mu, Sigma)
        g = r.pdf(pos)

        self.G = np.expand_dims(fft2(g), axis=0)  # using same desired response for all channels, P is organized (channels, height, width)

        A = self.G * np.conj(F)
        B = F * np.conj(F)

        image_center = (self.region.xpos + self.region_center[1], self.region.ypos + self.region_center[0])
        for angle in np.arange(-20,20,5):
            img_tmp = rotateImage(image, angle, image_center) # Rotate
            features = [img_tmp[...,i] for i in range(img_tmp.shape[-1])]
            f = self.get_patch(features)
            f = self.preprocess_data(f)
            F = fft2(f)
            A += self.G * np.conj(F)
            B += F * np.conj(F)                
        
        self.A = A
        self.B = B
        self.H_conj = self.A / (self.B + self.epsilon)

        if self.save_img and self.frame % self.save_frame == 0:
            plot(image, g, self.search_region, "{0}_{1}".format(self.name, self.frame))

    def detect(self, image):
        self.frame += 1
        features = [image[...,i] for i in range(image.shape[-1])]
        assert len(features) == 3, print(len(features))
        f = self.get_patch(features)
        f = self.preprocess_data(f)
        F = fft2(f)
        R = F * self.H_conj
        responses = ifft2(R)
        response = responses.sum(axis=0)  # .real
        r, c = np.unravel_index(np.argmax(response), response.shape)
        if self.save_img and self.frame % self.save_frame == 0:
            plot(image, response, self.search_region, "{0}_{1}".format(self.name, self.frame))

        # Keep for visualisation
        self.last_response = response

        r_offset = r - self.region_center[0]
        c_offset = c - self.region_center[1]

        self.region.xpos += c_offset
        self.region.ypos += r_offset

        self.search_region.xpos += c_offset
        self.search_region.ypos += r_offset
        return self.region
    
    def update(self, image):
        features = [image[...,i] for i in range(image.shape[-1])]
        assert len(features) == 3, print(len(features))
        f = self.get_patch(features)
        f = self.preprocess_data(f)
        F = fft2(f)
        self.A = self.learning_rate * self.G * np.conj(F) + (1-self.learning_rate) * self.A
        self.B = self.learning_rate * F * np.conj(F) + (1-self.learning_rate) * self.B

        self.H_conj = self.A / (self.B + self.epsilon)