# Template for lab02 task 2

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

FILENAME = 'image2.jpg'

if __name__ == '__main__':
    # Read image with OpenCV and convert to grayscale
    # Initialize SIFT detector
    
    # FROM TASK 1:
    # 1. Read image
    img1 = cv2.imread(FILENAME)
    # 2. Convert image to grayscale
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # 3. Initialize SIFT detector
    sd = SiftDetector()
    # 4. Detect SIFT features
    kp1, des1 = sd.detector.detectAndCompute(img1,None)
    # 5. Visualize detected features
    kp1_img = cv2.drawKeypoints(img1, kp1, None)
    #cv2.imwrite('task2a_result.jpg', kp1_img)
    kp1_default_len = len(kp1)
    print('{}: {} keypoints (default)'.format(FILENAME, kp1_default_len))

    #######################################################
    # TASK 2: rescale to 110%
    nrow,ncol = img1.shape
    img2 = cv2.resize(img1, (int(ncol*1.1), int(nrow*1.1)) )

    # TASK 2b: keypoints of rescaled image (want ~1/4 of unscaled)
    params = {}
    params["n_features"]=0              # adjusted
    params["n_octave_layers"]=3
    params["contrast_threshold"]=0.1325   # adjusted for Task 1 conditions
    params["edge_threshold"]=10
    params["sigma"]=1.6
    kp2, des2 = sd.get_detector(params).detectAndCompute(img2,None)
    kp2_img = cv2.drawKeypoints(img2, kp2, None)
    #cv2.imwrite('task2b_result.jpg', kp2_img)
    kp2_new_len = len(kp2)
    print('{}: {} keypoints ({:.2f}%)'.format(FILENAME, kp2_new_len, 
                                              kp2_new_len/kp1_default_len*100))

    # brute force matching 
    # * create BFMatcher object
    # * https://docs.opencv.org/3.4/da/df5/tutorial_py_sift_intro.html
    # * https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html
    # * https://docs.opencv.org/3.4/d3/da1/classcv_1_1BFMatcher.html#ac6418c6f87e0e12a88979ea57980c020

    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=False)
    # * Match descriptors.
    matches = bf.match(des1,des2)
    # * Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)

    # * Draw first 10 matches.
    kp3_img = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10], None, flags=2)
    cv2.imwrite('task2c_result.jpg', kp3_img)

    # visual inspection
    print("Task 2d: Results look the same. Suggests that SIFT can handle scaling, or at least for examples "
          "like this one where the scaling is evenly applied to both directions.")

    # Store SIFT keypoints of original image in a Numpy array
    #kp = sd.detector.detect(img,None)
    #kp_np = np.array(kp)
    
    # center of image points. 'img_center' is in (Y, X) order. #(Skip this step for Task2)
    
    # Degrees with which to rotate image. #(Skip this step for Task2)

    # Rotate image #(For Task2 , rescale the image)
        
    # Compute SIFT features for rotated image #( For task2, rescaled image)
        
    # BFMatcher with default params

    # Apply ratio test

    # cv2.drawMatchesKnn to draw matches