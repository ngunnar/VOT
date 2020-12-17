#!/usr/bin/env python3

import cv2

import numpy as np

from cvl.dataset import OnlineTrackingBenchmark
from cvl.deep_mosse import DeepTracker

dataset_path = "Mini-OTB"

SHOW_TRACKING = True
SAVE_IMAGES = False
SEQUENCE_IDX = 0
feature_level = 3 #[0, 3, 6]
search_size = 1#1.0
learning_rate = 0.01
sigma = 2/search_size
name="deep"
save_frame = 1

if __name__ == "__main__":

    dataset = OnlineTrackingBenchmark(dataset_path)

    a_seq = dataset[SEQUENCE_IDX]

    if SHOW_TRACKING:
        cv2.namedWindow("tracker")
  
    tracker = DeepTracker(feature_level=feature_level,
                          search_size = search_size,
                          learning_rate = learning_rate,
                          sigma=sigma,
                          save_img=SAVE_IMAGES,
                          save_frame=save_frame,
                          name=name)

    for frame_idx, frame in enumerate(a_seq):
        print(f"{frame_idx} / {len(a_seq)}")
        image = frame['image']

        if frame_idx == 0:
            bbox = frame['bounding_box']
            if bbox.width % 2 == 0:
                bbox.width += 1

            if bbox.height % 2 == 0:
                bbox.height += 1

            current_position = bbox
            tracker.start(image, bbox)
        else:
            tracker.detect(image)
            tracker.update(image)

        if SHOW_TRACKING:
            bbox = tracker.region
            pt0 = (bbox.xpos, bbox.ypos)
            pt1 = (bbox.xpos + bbox.width, bbox.ypos + bbox.height)
            image_color = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.rectangle(image_color, pt0, pt1, color=(0, 255, 0), thickness=3)
            cv2.imshow("tracker", image_color)
            cv2.waitKey(0)
