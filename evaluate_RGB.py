#!/usr/bin/env python3

import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib

from cvl.dataset import OnlineTrackingBenchmark
from cvl.trackers import NCCTracker
from cvl.rgb_mosse import MultiMosseTracker

from evaluate import evaluate


if __name__ == "__main__":
    evaluate(MultiMosseTracker, 'rgb')
