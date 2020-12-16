#!/usr/bin/env python3

import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib

from cvl.dataset import OnlineTrackingBenchmark
from cvl.trackers import NCCTracker
from cvl.grayscale_mosse import GrayscaleMosseTracker


def compute_iou(frame_data, tracked_box):
    gt_box = frame_data['bounding_box']
    union = gt_box.union_box(tracked_box)
    intersection = gt_box.intersection_box(tracked_box)
    iou = intersection.area() / union.area()

    return iou

def compute_ar(iou_list, threshold=0.1, threshold_frames=10):
    # get the index where the iou is below the threshold for threshold_frames consecutive values
    idx_low = len(iou_list)
    idx_count = 0
    for i, iou in enumerate(iou_list):
        if iou < threshold:
            idx_low = i
            idx_count += 1
        else:
            idx_low = len(iou_list)
            idx_count = 0
        if idx_count >= threshold_frames:
            break
    idx_low -= threshold_frames

    # compute accuracy within the successfully tracked period
    accuracy_ = np.asarray(iou_list[:idx_low]).mean()
    robustness_ = idx_low / len(iou_list)

    return accuracy_, robustness_


if __name__ == "__main__":

    # initialise the dataset
    dataset_path = "Mini-OTB"
    dataset = OnlineTrackingBenchmark(dataset_path)

    # allocation
    acc_list = []
    rob_list = []

    # for all sequences in the dataset
    for seq_id, a_seq in enumerate(dataset):
        # allocate
        iou = []

        # initialise the tracker
        tracker = GrayscaleMosseTracker()

        # initialise progress bar
        process_desc = "Seq {:}/{:}, '{:s}'"
        progress_bar = tqdm(initial=0, leave=True, total=len(a_seq),
                            desc=process_desc.format(int(seq_id) + 1, len(dataset), a_seq.sequence_name),
                            position=0)
        for frame_idx, frame in enumerate(a_seq):
            image_color = frame['image']
            image = np.sum(image_color, 2) / 3

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

            # compute iou
            iou.append(compute_iou(frame, tracker.region))

            # Update train bar
            progress_bar.update(1)
        progress_bar.close()

        # compute accuracy and robustness
        acc, rob = compute_ar(iou)
        acc_list.append(acc)
        rob_list.append(rob)

    # average accuracy and robustness
    acc_avg = np.asarray(acc_list).mean()
    rob_avg = np.asarray(rob_list).mean()

    # plot accuacry / robustness plot
    plt.figure(dpi=300)
    plt.axes().set_aspect('equal')
    plt.scatter(rob_list, acc_list, label='individual')
    for i, (r, a) in enumerate(zip(rob_list, acc_list)):
        plt.annotate(i, (r, a))

    plt.scatter(rob_avg, acc_avg, color='r', marker='x', label='avg')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('Robustness')
    plt.ylabel('Accuracy')
    plt.title('AR raw plot')
    plt.legend(loc='lower right')
    path = os.path.join(os.getcwd(), 'results')
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(os.path.join(path, 'grayscale.png'))