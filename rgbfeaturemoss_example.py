#!/usr/bin/env python3

import cv2

from cvl.dataset import OnlineTrackingBenchmark
from cvl.rgb_mosse_feature import MultiFeatureMosseTracker
from skimage.feature import hog

dataset_path = "Mini-OTB"

SHOW_TRACKING = True
SEQUENCE_IDX = 4


if __name__ == "__main__":

    dataset = OnlineTrackingBenchmark(dataset_path)
    a_seq = dataset[SEQUENCE_IDX]
    
    if SHOW_TRACKING:
        cv2.namedWindow("tracker") 

    tracker = MultiFeatureMosseTracker(save_img=True)

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
            score = tracker.detect(image) 
            #if score > 0.05: 
            tracker.update(image)

        if SHOW_TRACKING:      
            bbox = tracker.region
            pt0 = (bbox.xpos, bbox.ypos)
            pt1 = (bbox.xpos + bbox.width, bbox.ypos + bbox.height)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.rectangle(image, pt0, pt1, color=(0, 255, 0), thickness=3)
            cv2.imshow("tracker", image)
            cv2.waitKey(0)
