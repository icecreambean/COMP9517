# Template for lab02 task 3

import cv2
import math
import numpy as np
import sys

class SiftDetector():
    def __init__(self, norm="L2", params=None):
        self.detector=self.get_detector(params)
        self.norm=norm

    def get_detector(self, params):
        if params is None:
            params={}
            params["n_features"]=0
            params["n_octave_layers"]=3
            params["contrast_threshold"]=0.04
            params["edge_threshold"]=10
            params["sigma"]=1.6

        detector = cv2.xfeatures2d.SIFT_create(
                nfeatures=params["n_features"],
                nOctaveLayers=params["n_octave_layers"],
                contrastThreshold=params["contrast_threshold"],
                edgeThreshold=params["edge_threshold"],
                sigma=params["sigma"])

        return detector

FILENAME = 'image1.jpg'

if __name__ == '__main__':
    # 1. Read image
    img = cv2.imread(FILENAME)
    # 2. Convert image to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 3. Initialize SIFT detector
    sd = SiftDetector()
    # 4. Detect SIFT features
    kp = sd.detector.detect(img,None)
    # 5. Visualize detected features
    kp_img = cv2.drawKeypoints(img, kp, None)
    cv2.imwrite('task1a_result.jpg', kp_img)
    # Print number of SIFT features detected
    kp_default_len = len(kp)
    print('{}: {} keypoints (default)'.format(FILENAME, kp_default_len))

    # TASK 1b
    params = {}
    params["n_features"]=0              # adjusted
    params["n_octave_layers"]=3
    params["contrast_threshold"]=0.0825   # adjusted
    params["edge_threshold"]=10
    params["sigma"]=1.6
    kp = sd.get_detector(params).detect(img,None)
    kp_img = cv2.drawKeypoints(img, kp, None)
    cv2.imwrite('task1b_result.jpg', kp_img)
    kp_new_len = len(kp)
    print('{}: {} keypoints ({:.2f}%)'.format(FILENAME, kp_new_len, 
                                              kp_new_len/kp_default_len*100))
