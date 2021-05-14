import numpy as np 

class FeatureExtractor(object):

    def __init__(self, **kwargs):
        pass 

    def get_feature(self, **kwargs):
        pass 

class ForexFeature(FeatureExtractor):

    def __init__(self):
        self.dimension = 200

    def get_feature(self, feature_history):
        return feature_history[-1][1]