import numpy as np
import matplotlib.pyplot as plt
import os
from skimage import color
from skimage.feature import hog
from sklearn import svm
#from sklearn.metrics import 

DIR = 'Individual_Component'



# load example image
#fn = 'item_00000000.pnm'
#fp = os.path.join(DIR,'test','test_positive','00000000', fn)
fn = 'highres_test2.png'
fp = os.path.join(DIR, fn)

print('Loading image...')
img = cv2.imread(fp)
print('Image loaded. Applying HOG+SVM...')

(rects, weights) = hog.detectMultiScale(img, winStride=(1, 1), 
                                        padding=(32, 32), scale=1.05)
print(rects)
print(weights)
print('Done. Drawing boxes...')
# draw the original bounding boxes
for (x, y, w, h) in rects:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0,0,255), 2)

# # apply non-maxima suppression to the bounding boxes using a
# # fairly large overlap threshold to try to maintain overlapping
# # boxes that are still people
# rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
# pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
# # draw the final bounding boxes
# for (xA, yA, xB, yB) in pick:
#     cv2.rectangle(img, (xA, yA), (xB, yB), (0, 255, 0), 2)

# show result
print('Done. Displaying result...')
plt.figure()
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('{}: #rects={}, #pick={}'.format(fn, len(rects), 0))#len(pick)))
plt.show()
print('Done. Program terminated.')