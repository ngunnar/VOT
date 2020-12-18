import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch

from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
from scipy.stats import multivariate_normal
from scipy.ndimage import rotate

from .image_io import crop_patch
from .utils import pre_process, rotateImage, plot, get_search_region
from .features import alexnetFeatures, AlexNetRefined
from .dataset import BoundingBox

########################################################################################################################
# Deep Mosse
########################################################################################################################
def to_torch_tensor(image):
    return torch.from_numpy(image.astype('float32'))[None,...]

class DeepTracker():
    def __init__(self, 
                feature_level = 3, #[0, 3, 6]
                search_size = 1.0,
                learning_rate = 0.05,
                save_img = False,
                sigma = 2.0,
                name="deep",
                save_frame = 10):
        
        assert feature_level in [0, 3, 6], "Only use feature maps after conv2d layers, {0}".format(feature_level)
        self.learning_rate = learning_rate 
        self.lambda_ = 1e-2
        self.sigma = sigma
        self.org_model = alexnetFeatures(pretrained=True)
        self.model = AlexNetRefined(self.org_model, feature_level)
        self.search_size = search_size
        self.save_img = save_img
        self.name = name
        self.save_frame = save_frame

    def get_patch_features(self, image):
        # Image is the first frame
        features = [image[...,i]/255 for i in range(image.shape[-1])]
        # Extract patches from image
        f = self.get_patch(features, self.search_region)
        f = np.asarray(f)
        f = np.moveaxis(f, 0, -1)
        f = cv2.resize(f, dsize=(244, 244), interpolation=cv2.INTER_CUBIC)
        f = np.moveaxis(f, -1, 0)
        input_torch = to_torch_tensor(np.asarray(f))
        output_features = self.model.forward(input_torch).detach().numpy()[0,...]
        f = np.asarray([pre_process(output_features[i,...]) for i in range(output_features.shape[0])])
        return f

    def get_patch(self, features, region):
        return [crop_patch(f, region) for f in features]

    def start(self, image, region):
        self.frame = 0
        # Region is the bounding box around target in first frame
        self.region = region
        self.region_shape = (region.height, region.width)
        self.region_center = (region.height // 2, region.width // 2)
        
        self.search_region = region.rescale(self.search_size,round_coordinates=True)
        self.search_region_shape = (self.search_region.height, self.search_region.width)
        self.search_region_center = (self.search_region.height // 2, self.search_region.width // 2)

        f = self.get_patch_features(image)
        F = fft2(f)
        # Create desired response
        mu = [f.shape[1]//2, f.shape[2]//2]

        Sigma = np.eye(2) * self.sigma ** 2
        x, y = np.mgrid[0:f.shape[1]:1,
               0:f.shape[2]:1]
        pos = np.dstack((x, y))
        r = multivariate_normal(mu, Sigma)
        g = r.pdf(pos)

        self.G = np.expand_dims(fft2(g), axis=0)  # using same desired response for all channels, P is organized (channels, height, width)

        A = self.G * np.conj(F)
        B = F * np.conj(F)        

        image_center = (self.region.xpos + self.region_center[1], self.region.ypos + self.region_center[0])
        for angle in np.arange(-20,20,5):
            img_tmp = rotateImage(image, angle, image_center) # Rotate
            for blur in range(1,10):
                img_tmp = cv2.blur(img_tmp, (blur,blur))
                f = self.get_patch_features(img_tmp)
                F = fft2(f)
                A += self.G * np.conj(F)
                B += F * np.conj(F)

        self.A = A
        self.B = B
        self.H_conj = self.A / (self.B + self.lambda_)

        if self.save_img and self.frame % self.save_frame == 0:
            response = cv2.resize(g.real, dsize=(self.search_region_shape[1],self.search_region_shape[0]), interpolation=cv2.INTER_CUBIC)
            plot(image, response, self.search_region, "{0}_{1}".format(self.name, self.frame))

    def detect(self, image):
        self.frame += 1
        f = self.get_patch_features(image)

        F = fft2(f)
        R = F * self.H_conj
        responses = ifft2(R)

        response = responses.sum(axis=0)  # .real
        response = cv2.resize(response.real, dsize=(self.search_region_shape[1],self.search_region_shape[0]), interpolation=cv2.INTER_CUBIC)
        r, c = np.unravel_index(np.argmax(response), response.shape)
        
        if self.save_img and self.frame % self.save_frame == 0:
            plot(image, response, self.search_region, "{0}_{1}".format(self.name, self.frame))

        # Keep for visualisation
        self.last_response = response

        r_offset = r - self.search_region_center[0]
        c_offset = c - self.search_region_center[1]
 
        self.region.xpos += c_offset
        self.region.ypos += r_offset
        
        self.search_region.xpos += c_offset
        self.search_region.ypos += r_offset
        return self.region
    
    def update(self, image):
        f = self.get_patch_features(image)
        F = fft2(f)

        self.A = self.learning_rate * self.G * np.conj(F) + (1-self.learning_rate) * self.A
        self.B = self.learning_rate * F * np.conj(F) + (1-self.learning_rate) * self.B

        self.H_conj = self.A / (self.B + self.lambda_)