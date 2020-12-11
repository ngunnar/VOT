#!/usr/bin/env python3

import cv2

import numpy as np

from cvl.dataset import OnlineTrackingBenchmark, BoundingBox
from cvl.rgb_mosse import MultiMosseTracker

dataset_path = "Mini-OTB"

SHOW_TRACKING = True
SEQUENCE_IDX = 4


if __name__ == "__main__":

    dataset = OnlineTrackingBenchmark(dataset_path)
    a_seq = dataset[SEQUENCE_IDX]
    
    if SHOW_TRACKING:
        cv2.namedWindow("tracker") 

    tracker = MultiMosseTracker()

    for frame_idx, frame in enumerate(a_seq):
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
            score = tracker.detect(features) 
            #if score > 0.05: 
            tracker.update(features)

        if SHOW_TRACKING:      
            bbox = tracker.region
            pt0 = (bbox.xpos, bbox.ypos)
            pt1 = (bbox.xpos + bbox.width, bbox.ypos + bbox.height)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.rectangle(image, pt0, pt1, color=(0, 255, 0), thickness=3)
            cv2.imshow("tracker", image)
            cv2.waitKey(0)
