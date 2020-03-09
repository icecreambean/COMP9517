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

# Rotate an image
#
# image: image to rotate
# x:     x-coordinate of point we wish to rotate around
# y:     y-coordinate of point we wish to rotate around
# angle: degrees to rotate image by
#
# Returns a rotated copy of the original image
def rotate(image, x, y, angle):
    # Source: https://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/
    (h, w) = image.shape[:2]
    #(cX, cY) = (w // 2, h // 2)
    cX = x
    cY = y

    # grab the rotation matrix (rotates anticlockwise), then grab the 
    # sine and cosine (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))


# Get coordinates of center point.
#
# image:  Image that will be rotated
# return: (x, y) coordinates of point at center of image
def get_img_center(image):
    # Source: https://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/
    # https://stackoverflow.com/questions/1535596/what-is-the-reason-for-having-in-python
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    return (cX, cY)

FILENAME = 'image1.jpg'

if __name__ == '__main__':
    # Read image with OpenCV and convert to grayscale
    # Initialize SIFT detector
    # Store SIFT keypoints of original image in a Numpy array
    
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
    #cv2.imwrite('task3a_result.jpg', kp1_img)
    kp1_default_len = len(kp1)
    print('{}: {} keypoints (default)'.format(FILENAME, kp1_default_len))

    #######################################################
    # center of image points. 'img_center' is in (Y, X) order. #(Skip this step for Task2)
    cx,cy = get_img_center(img1)

    # Degrees with which to rotate image. #(Skip this step for Task2)
    angle1 = 90 # (90: anticlockwise) (-90: clockwise)

    # Rotate image #(For Task2 , rescale the image)
    img2 = rotate(img1, cx, cy, angle1)
    #cv2.imwrite('task3aii_result.jpg', img2)

    #(tweaked version of copy-pasted code from Task 2)
    # Compute SIFT features for rotated image #( For task2, rescaled image)
    params = {}
    params["n_features"]=0              # adjusted
    params["n_octave_layers"]=3
    params["contrast_threshold"]=0.0825   # adjusted for Task 1 conditions
    params["edge_threshold"]=10
    params["sigma"]=1.6
    kp2, des2 = sd.get_detector(params).detectAndCompute(img2,None)
    kp2_img = cv2.drawKeypoints(img2, kp2, None)
    #cv2.imwrite('task2b_result.jpg', kp2_img)
    kp2_new_len = len(kp2)
    print('{}: {} keypoints ({:.2f}%)'.format(FILENAME, kp2_new_len, 
                                              kp2_new_len/kp1_default_len*100))

    # BFMatcher with default params
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(des1,des2, k=2)   # k=2 for ratio test

    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])
    # cv2.drawMatchesKnn to draw matches
    # cv2.drawMatchesKnn expects list of lists as matches.
    kp3_img = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)
    cv2.imwrite('task3c_result.jpg', kp3_img)


        
