import numpy as np
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
from scipy.stats import multivariate_normal
from scipy.ndimage import rotate
from .image_io import crop_patch
import matplotlib.pyplot as plt
from .utils import pre_process, rotateImage, plot
from .features import alexnetFeatures, AlexNetRefined
import cv2

########################################################################################################################
# Deep Mosse
########################################################################################################################

import torch
def to_torch_tensor(image):
    return torch.from_numpy(image.astype('float32'))[None,...]

from cvl.dataset import BoundingBox
def get_search_region(region, ratio):
    region_shape = (region.height, region.width)
    search_shape = tuple([int(np.round(ratio*x)) for x in region_shape])
    xpos = int(np.round(region.xpos - (search_shape[1]-region_shape[1])/2))
    ypos = int(np.round(region.ypos - (search_shape[0]-region_shape[0])/2))
    search_region = BoundingBox('tl-size', xpos, ypos, search_shape[1], search_shape[0])
    return search_region

#
#search_size = 1.0
#feature_deep = 3#[0, 3, 6, 8, 10]
class DeepTracker():
    def __init__(self, 
                feature_level = 3, 
                search_size = 3.0,
                learning_rate = 0.05,
                save_img = False):
        self.learning_rate = learning_rate 
        self.lambda_ = 1e-1
        self.sigma = 1.0
        self.org_model = alexnetFeatures(pretrained=True)
        self.model = AlexNetRefined(self.org_model, feature_level)
        self.search_size = search_size
        self.save_img = save_img

    def get_patch_features(self, image):
        # Image is the first frame
        features = [image[...,i]/255 for i in range(image.shape[-1])]
        # Extract patches from image
        p = self.get_patch(features, self.search_region)

        input_torch = to_torch_tensor(np.asarray(p))
        output_features = self.model.forward(input_torch).detach().numpy()[0,...]
        p_prepocessed = np.asarray([pre_process(output_features[i,...]) for i in range(output_features.shape[0])])
        return p_prepocessed, p

    def get_patch(self, features, region):
        return [crop_patch(c, region) for c in features]

    def preprocess_data(self, features):
        return np.asarray([pre_process(c) for c in features])

    def start(self, image, region):
        self.frame = 0
        # Region is the bounding box around target in first frame
        self.region = region
        self.region_shape = (region.height, region.width)
        self.region_center = (region.height // 2, region.width // 2)
        
        self.search_region = get_search_region(region, self.search_size)
        self.search_region_shape = (self.search_region.height, self.search_region.width)
        self.search_region_center = (self.search_region.height // 2, self.search_region.width // 2)

        f_prepocessed, _ = self.get_patch_features(image)
        F = fft2(f_prepocessed)
        # Create desired response
        mu = [f_prepocessed.shape[1]//2, f_prepocessed.shape[2]//2]

        covariance = [[self.sigma ** 2, 0], [0, self.sigma ** 2]]
        x, y = np.mgrid[0:f_prepocessed.shape[1]:1,
               0:f_prepocessed.shape[2]:1]
        pos = np.dstack((x, y))
        r = multivariate_normal(mu, covariance)
        c = r.pdf(pos)

        self.C = np.expand_dims(fft2(c), axis=0)  # using same desired response for all channels, P is organized (channels, height, width)

        A = self.C * np.conj(F)
        B = F * np.conj(F)
        
        for _ in range(100):
            f_prepocessed, _ = self.get_patch_features(image + 0.1 * image.std() * np.random.random(image.shape))
            F = fft2(f_prepocessed)
            A += self.C * np.conj(F)
            B += F * np.conj(F)        
        
        image_center = (self.region.xpos + self.region_center[0], self.region.ypos + self.region_center[1])
        for angle in np.arange(-20,20,5):
            img_tmp = rotateImage(image, angle, image_center) # Rotate
            img_tmp = img_tmp + 0.5 * img_tmp.std() * np.random.random(img_tmp.shape) # Add noise
            f_prepocessed, _ = self.get_patch_features(img_tmp)
            F = fft2(f_prepocessed)
            A += self.C * np.conj(F)
            B += F * np.conj(F)

        self.A = A
        self.B = B
        self.H_conj = self.A / (self.B + self.lambda_)

        if self.save_img and self.frame % 10 == 0:
            response = cv2.resize(c.real, dsize=(self.search_region_shape[1],self.search_region_shape[0]), interpolation=cv2.INTER_CUBIC)
            plot(image, response, self.search_region, "deep_{0}".format(self.frame))

    def detect(self, image):
        self.frame += 1
        f_prepocessed, _ = self.get_patch_features(image)

        F = fft2(f_prepocessed)
        R = F * self.H_conj
        responses = ifft2(R)

        response = responses.sum(axis=0)  # .real
        response = cv2.resize(response.real, dsize=(self.search_region_shape[1],self.search_region_shape[0]), interpolation=cv2.INTER_CUBIC)
        r, c = np.unravel_index(np.argmax(response), response.shape)
        
        if self.save_img and self.frame % 10 == 0:
            plot(image, response, self.search_region, "deep_{0}".format(self.frame))

        # Keep for visualisation
        self.last_response = response

        r_offset = r - self.search_region_center[0]
        c_offset = c - self.search_region_center[1]

        #xpos_old = self.region.xpos
        #ypos_old = self.region.ypos
        self.region.xpos += c_offset
        self.region.ypos += r_offset
        
        self.search_region.xpos += c_offset
        self.search_region.ypos += r_offset
        return self.region
    
    def update(self, image):
        f_prepocessed, _ = self.get_patch_features(image)
        F = fft2(f_prepocessed)

        self.A = self.learning_rate * self.C * np.conj(F) + (1-self.learning_rate) * self.A
        self.B = self.learning_rate * F * np.conj(F) + (1-self.learning_rate) * self.B

        self.H_conj = self.A / (self.B + self.lambda_)