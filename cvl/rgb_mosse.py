import numpy as np
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
from scipy.stats import multivariate_normal
from scipy.ndimage import rotate
from .image_io import crop_patch
import matplotlib.pyplot as plt
from .utils import pre_process, rotateImage

########################################################################################################################
# RGB Mosse
########################################################################################################################

class MultiMosseTracker():
    def __init__(self, learning_rate = 0.125):
        self.learning_rate = learning_rate 
        self.lambda_ = 1e-5
        self.sigma = 2.0

    def get_patch(self, features):
        region = self.region
        return [crop_patch(c, region) for c in features]

    def preprocess_data(self, features):
        return np.asarray([pre_process(c) for c in features])

    def start(self, features, region):
        assert len(features) == 3, print(len(features))
        # Image is the first frame
        # Region is the bounding box around target in first frame
        self.region = region
        self.region_shape = (region.height, region.width)
        self.region_center = (region.height // 2, region.width // 2)
        

        # Extract patches from image
        p = self.get_patch(features)
        p_prepocessed = self.preprocess_data(p)
        P = fft2(p_prepocessed)

        # Create desired response
        mu = [self.region_center[0], self.region_center[1]]
        # mu = [0, 0]
        covariance = [[self.sigma ** 2, 0], [0, self.sigma ** 2]]
        x, y = np.mgrid[0:region.height:1,
               0:region.width:1]
        pos = np.dstack((x, y))
        r = multivariate_normal(mu, covariance)
        c = r.pdf(pos)

        self.C = np.expand_dims(fft2(c),
                                axis=0)  # using same desired response for all channels, P is organized (channels, height, width)

        A = np.conj(self.C) * P
        B = np.conj(P) * P

        # TODO train on augumented data
        # Rotate
        '''
        image_center = (self.region.xpos + self.region_center[0], self.region.ypos + self.region_center[1])
        for _ in range(100):
            p = self.get_patch([f + 0.2 * f.std() * np.random.random(f.shape) for f in features])
            p_prepocessed = self.preprocess_data(p)
            P = fft2(p_prepocessed)
            A += np.conj(self.C) * P
            B += np.conj(P) * P
        '''
        '''
        for angle in np.arange(-1,1,5):
            f_tmp = [rotateImage(f_, angle, image_center) for f_ in features] # Rotate
            #import matplotlib.pyplot as plt
            #plt.imshow(np.moveaxis(np.asarray(f_tmp), 0, -1))
            #plt.plot(image_center[0],image_center[1], 'ro')
            #plt.show()
            f_tmp = [f_ + 0.5 * f_.std() * np.random.random(f_.shape) for f_ in f_tmp] # Add noise
            p_tmp = self.get_patch(f_tmp)
            p_prepocessed = self.preprocess_data(p_tmp)
            P = fft2(p_prepocessed)
            A += np.conj(self.C) * P
            B += np.conj(P) * P
        '''
        self.A = A
        self.B = B
        self.M = self.A / (self.B + self.lambda_)

        if True:
            max_val = np.max(c)
            max_pos = np.where(c == max_val)
            cs = np.asarray([c for _ in range(3)])
            self.plot(p_prepocessed, cs, c, max_pos[1], max_pos[0])

    def detect(self, features):
        assert len(features) == 3, print(len(features))
        p = self.get_patch(features)
        p_prepocessed = self.preprocess_data(p)
        P = fft2(p_prepocessed)
        R = np.conj(self.M) * P
        responses = ifft2(R)
        response = responses.sum(axis=0)  # .real
        r, c = np.unravel_index(np.argmax(response), response.shape)
        print("Score {0}".format(response[r, c]))
        if False:
            self.plot(p, responses, response, c, r)

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
        print("Old: {0}, {1}".format(xpos_old, ypos_old))
        print("New: {0}, {1}".format(self.region.xpos, self.region.ypos))
        print("P: {0}, {1}".format(r, c))
        return self.region
    def update(self, features):
        p = self.get_patch(features)
        p_prepocessed = self.preprocess_data(p)
        P = fft2(p_prepocessed)
        self.A = self.learning_rate * np.conj(self.C) * P  + (1 - self.learning_rate)*self.A
        self.B = self.learning_rate * np.conj(P) * P + (1 - self.learning_rate)*self.B
        self.M = self.A/(self.B + self.lambda_)

    def plot(self, p, responses, response, c, r):
        import matplotlib.pyplot as plt
        _, axs = plt.subplots(3, 3)
        axs = axs.flatten()
        ax_template = axs[0]
        axs[1].imshow(np.zeros(p[0, ...].shape))
        ax_response = axs[2]

        ax_filter1 = axs[3]
        ax_filter2 = axs[4]
        ax_filter3 = axs[5]

        ax_resp1 = axs[6]
        ax_resp2 = axs[7]
        ax_resp3 = axs[8]

        ax_template.imshow(np.moveaxis(p, 0, -1))
        ax_template.set_title("Template image x")

        ax_response.imshow(np.real(response))
        ax_response.set_title("Responses")
        ax_response.plot(c, r, 'bo')

        filters = np.real(ifft2(np.conjugate(self.M)))
        ax_filter1.imshow(filters[0, ...], cmap=plt.get_cmap('gray'))
        ax_filter1.set_title("Filter f1")

        ax_filter2.imshow(filters[1, ...], cmap=plt.get_cmap('gray'))
        ax_filter2.set_title("Filter f2")

        ax_filter3.imshow(filters[2, ...], cmap=plt.get_cmap('gray'))
        ax_filter3.set_title("Filter f3")

        ax_resp1.imshow(np.real(responses[0, ...]), cmap=plt.get_cmap('gray'))
        ax_resp1.plot(c, r, 'bo')
        ax_resp2.imshow(np.real(responses[0, ...]), cmap=plt.get_cmap('gray'))
        ax_resp2.plot(c, r, 'bo')
        ax_resp3.imshow(np.real(responses[0, ...]), cmap=plt.get_cmap('gray'))
        ax_resp3.plot(c, r, 'bo')

        plt.show()