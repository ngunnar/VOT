#!/usr/bin/env python3

from cvl.deep_mosse import DeepTracker
from evaluate import evaluate


if __name__ == "__main__":
    evaluate(DeepTracker, 'deep0_2', feature_level=0, search_size=2)
    evaluate(DeepTracker, 'deep3_2', feature_level=3, search_size=2)
    evaluate(DeepTracker, 'deep6_2', feature_level=6, search_size=2)
