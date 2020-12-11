import numpy as np
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
from scipy.stats import multivariate_normal
from scipy.ndimage import rotate
from .image_io import crop_patch
import matplotlib.pyplot as plt
from .utils import pre_process, rotateImage
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

class DeepTracker():
    def __init__(self, feature_level, search_size, learning_rate = 0.125):
        self.learning_rate = learning_rate 
        self.lambda_ = 1e-5
        self.sigma = 1.0
        self.org_model = alexnetFeatures(pretrained=True)
        self.model = AlexNetRefined(self.org_model, feature_level)
        self.search_size = search_size

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

        # Region is the bounding box around target in first frame
        self.region = region
        self.region_shape = (region.height, region.width)
        self.region_center = (region.height // 2, region.width // 2)
        
        self.search_region = get_search_region(region, self.search_size)
        self.search_region_shape = (self.search_region.height, self.search_region.width)
        self.search_region_center = (self.search_region.height // 2, self.search_region.width // 2)

        p_prepocessed, _ = self.get_patch_features(image)
        P = fft2(p_prepocessed)
        # Create desired response
        mu = [p_prepocessed.shape[1]//2, p_prepocessed.shape[2]//2]

        covariance = [[self.sigma ** 2, 0], [0, self.sigma ** 2]]
        x, y = np.mgrid[0:p_prepocessed.shape[1]:1,
               0:p_prepocessed.shape[2]:1]
        pos = np.dstack((x, y))
        r = multivariate_normal(mu, covariance)
        c = r.pdf(pos)

        self.C = np.expand_dims(fft2(c), axis=0)  # using same desired response for all channels, P is organized (channels, height, width)

        A = np.conj(self.C) * P
        B = np.conj(P) * P

        # TODO train on augumented data
        self.A = A
        self.B = B
        self.M = self.A / (self.B + self.lambda_)

        if False:
            max_val = np.max(c)
            max_pos = np.where(c == max_val)
            cs = np.asarray([c for _ in range(3)])
            self.plot(image, c)

    def detect(self, image):
        p_prepocessed, p = self.get_patch_features(image)

        P = fft2(p_prepocessed)
        R = np.conj(self.M) * P
        responses = ifft2(R)
        response = responses.sum(axis=0)  # .real
        response = cv2.resize(response.real, dsize=(self.search_region_shape[1],self.search_region_shape[0]), interpolation=cv2.INTER_CUBIC)
        r, c = np.unravel_index(np.argmax(response), response.shape)
        print("Score {0}".format(response[r, c]))
        if False:
            self.plot(image, response)

        # Keep for visualisation
        self.last_response = response

        r_offset = r - self.search_region_center[0]
        c_offset = c - self.search_region_center[1]

        xpos_old = self.region.xpos
        ypos_old = self.region.ypos
        self.region.xpos += c_offset
        self.region.ypos += r_offset
        
        self.search_region.xpos += c_offset
        self.search_region.ypos += r_offset
        print("Old: {0}, {1}".format(xpos_old, ypos_old))
        print("New: {0}, {1}".format(self.region.xpos, self.region.ypos))
        print("P: {0}, {1}".format(r, c))
        self.plot(image, response)
        return self.region
    
    def update(self, image):
        p_prepocessed = self.get_patch_features(image)
        P = fft2(p_prepocessed)
        self.A = self.learning_rate * np.conj(self.C) * P  + (1 - self.learning_rate)*self.A
        self.B = self.learning_rate * np.conj(P) * P + (1 - self.learning_rate)*self.B
        self.M = self.A/(self.B + self.lambda_)

    def plot(self, image, response):
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
        xmin = self.search_region.xpos
        ymin = self.search_region.ypos
        xmax = xmin + self.search_region.width
        ymax = ymin + self.search_region.height
        print(xmin, xmax, ymin,ymax, image.shape, response.shape)
        if xmin < 0:
            xmin = 0
            dx = -xmin
        else:
            xmin = xmin
            dx = 0
        
        if ymin < 0:
            ymin = 0
            dy = -ymin
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

        r[ymin:ymax, xmin:xmax] = response[dy:dyy, dx:dxx]
        
        fig, ax = plt.subplots(1, 1)

        ax.imshow(image)
        cb = ax.contourf(x,y,r, 15, cmap=mycmap)

        plt.show()