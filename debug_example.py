#!/usr/bin/env python3

import cv2

import numpy as np

from cvl.dataset import OnlineTrackingBenchmark, BoundingBox
from cvl.rgb_mosse import MultiMosseTracker

SHOW_TRACKING = True

if __name__ == "__main__":

    a_seq = {}
    j_x = 0
    j_y = 0
    for i in range(20):
        image = np.zeros((500,500,3)).astype('float32')
        image[j_x:40+j_x,j_y:40+j_y,:] = 1.0
        a_seq[i] = {'image': image,
                'bounding_box': BoundingBox('tl-size', j_x-20, j_y-20, 80, 80)}
        j_x += 5
        j_y += 5   
    
    for i in range(20,40):
        image = np.zeros((500,500,3)).astype('float32')
        image[j_x:40+j_x,j_y:40+j_y,:] = 1.0
        a_seq[i] = {'image': image,
                'bounding_box': BoundingBox('tl-size', j_x-20, j_y-20, 80, 80)}
        j_x -= 5
        j_y += 5
    
    for i in range(40,60):
        image = np.zeros((500,500,3)).astype('float32')
        image[j_x:40+j_x,j_y:40+j_y,:] = 1.0
        a_seq[i] = {'image': image,
                'bounding_box': BoundingBox('tl-size', j_x-20, j_y-20, 80, 80)}
        j_y -= 5
    
    if SHOW_TRACKING:
        cv2.namedWindow("tracker") 

    tracker = MultiMosseTracker()

    for i, (frame_idx, frame) in enumerate(a_seq.items()):
        print(f"{frame_idx} / {len(a_seq)}")
        image = frame['image']
        features = [image[...,i] for i in range(image.shape[-1])]
        #image = np.sum(image_color, 2) / 3

        if frame_idx == 0:
            bbox = frame['bounding_box']
            if bbox.width % 2 == 0:
                bbox.width += 1

            if bbox.height % 2 == 0:
                bbox.height += 1

            current_position = bbox
            tracker.start(features, bbox)
        else:
            tracker.detect(features)         
            tracker.update(features)

        if SHOW_TRACKING:      
            bbox = tracker.region
            pt0 = (bbox.xpos, bbox.ypos)
            pt1 = (bbox.xpos + bbox.width, bbox.ypos + bbox.height)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.rectangle(image, pt0, pt1, color=(0, 255, 0), thickness=3)
            cv2.imshow("tracker", image)
            cv2.waitKey(0)
