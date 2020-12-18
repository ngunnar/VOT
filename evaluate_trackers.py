#!/usr/bin/env python3

from cvl.trackers import NCCTracker
from cvl.grayscale_mosse import GrayscaleMosseTracker
from cvl.rgb_mosse_feature import MultiFeatureMosseTracker
from cvl.rgb_mosse import MultiMosseTracker
from cvl.deep_mosse import DeepTracker
from evaluate import evaluate


if __name__ == "__main__":
    evaluate(DeepTracker, 'deep0', feature_level=0)
    evaluate(DeepTracker, 'deep3', feature_level=3)
    evaluate(DeepTracker, 'deep6', feature_level=6)
    evaluate(GrayscaleMosseTracker, 'grayscale')
    evaluate(MultiFeatureMosseTracker, 'rbg_hog')
    evaluate(MultiMosseTracker, 'rgb')
