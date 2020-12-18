#!/usr/bin/env python3

from cvl.trackers import NCCTracker
from cvl.grayscale_mosse import GrayscaleMosseTracker
from cvl.rgb_mosse_feature import MultiFeatureMosseTracker
from cvl.rgb_mosse import MultiMosseTracker
from cvl.deep_mosse import DeepTracker
from evaluate import evaluate


if __name__ == "__main__":
    search_size = 2
    evaluate(DeepTracker, 'deep0_{0}'.format(search_size), feature_level=0, search_size = search_size)
    evaluate(DeepTracker, 'deep3_{0}'.format(search_size), feature_level=3, search_size = search_size)
    evaluate(DeepTracker, 'deep6_{0}'.format(search_size), feature_level=6, search_size = search_size)
    evaluate(GrayscaleMosseTracker, 'grayscale_{0}'.format(search_size), search_size=search_size)
    evaluate(MultiFeatureMosseTracker, 'rbg_hog_{0}'.format(search_size), search_size=search_size)
    evaluate(MultiMosseTracker, 'rgb_{0}'.format(search_size), search_size = search_size)
