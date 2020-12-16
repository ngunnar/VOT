import numpy as np
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
from scipy.stats import multivariate_normal
from scipy.ndimage import rotate
from .image_io import crop_patch
import matplotlib.pyplot as plt
from .utils import pre_process, rotateImage, plot

########################################################################################################################
# RGB Mosse
########################################################################################################################

class MultiMosseTracker():
    def __init__(self, learning_rate = 0.125, plot_img=False):
        self.learning_rate = learning_rate 
        self.epsilon = 1e-5
        self.sigma = 2.0
        self.plot_img = plot_img

    def get_patch(self, features):
        region = self.region
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
        

        # Extract patches from image
        f = self.get_patch(features)
        f_prepocessed = self.preprocess_data(f)
        F = fft2(f_prepocessed)

        # Create desired response
        mu = [self.region_center[0], self.region_center[1]]
        # mu = [0, 0]
        covariance = [[self.sigma ** 2, 0], [0, self.sigma ** 2]]
        x, y = np.mgrid[0:region.height:1,
               0:region.width:1]
        pos = np.dstack((x, y))
        r = multivariate_normal(mu, covariance)
        g = r.pdf(pos)

        self.G = np.expand_dims(fft2(g), axis=0)  # using same desired response for all channels, P is organized (channels, height, width)

        A = self.G * np.conj(F)
        B = F * np.conj(F)

        # TODO train on augumented data
        # Rotate
        for _ in range(100):
            f = self.get_patch([f + 0.2 * f.std() * np.random.random(f.shape) for f in features])
            f_prepocessed = self.preprocess_data(f)
            F = fft2(f_prepocessed)
            A += self.G * np.conj(F)
            B += F * np.conj(F) 
        
        
        image_center = (self.region.xpos + self.region_center[0], self.region.ypos + self.region_center[1])
        for angle in np.arange(-20,20,5):
            img_tmp = rotateImage(image, angle, image_center) # Rotate
            img_tmp = img_tmp + 0.5 * img_tmp.std() * np.random.random(img_tmp.shape) # Add noise
            f_tmp = [img_tmp[...,i] for i in range(img_tmp.shape[-1])]
            f_tmp = self.get_patch(f_tmp)
            f_prepocessed = self.preprocess_data(f_tmp)
            F = fft2(f_prepocessed)
            A += self.G * np.conj(F)
            B += F * np.conj(F) 
        
        self.A = A
        self.B = B
        self.H_conj = self.A / (self.B + self.epsilon)

        if self.plot_img and self.frame % 10 == 0:
            plot(image, g, self.region, "rbg_{0}".format(self.frame))

    def detect(self, image):
        self.frame += 1
        features = [image[...,i] for i in range(image.shape[-1])]
        assert len(features) == 3, print(len(features))
        f = self.get_patch(features)
        f_prepocessed = self.preprocess_data(f)
        F = fft2(f_prepocessed)
        R = F * self.H_conj
        responses = ifft2(R)
        response = responses.sum(axis=0)  # .real
        r, c = np.unravel_index(np.argmax(response), response.shape)
        if self.plot_img and self.frame % 10 == 0:
            plot(image, response, self.region, "rbg_{0}".format(self.frame))

        # Keep for visualisation
        self.last_response = response

        r_offset = r - self.region_center[0]
        c_offset = c - self.region_center[1]

        # r_offset = np.mod(r + self.region_center[0], self.region.height) - self.region_center[0]
        # c_offset = np.mod(c + self.region_center[1], self.region.width) - self.region_center[1]

        xpos_old = self.region.xpos
        ypos_old = self.region.ypos
        self.region.xpos += c_offset
        self.region.ypos += r_offset
        return self.region
    
    def update(self, image):
        features = [image[...,i] for i in range(image.shape[-1])]
        f = self.get_patch(features)
        f_prepocessed = self.preprocess_data(f)
        F = fft2(f_prepocessed)
        self.A = self.learning_rate * self.G * np.conj(P) + (1-self.learning_rate) * self.A
        self.B = self.learning_rate * P * np.conj(P) + (1-self.learning_rate) * self.B

        self.H_conj = self.A / (self.B + self.epsilon)