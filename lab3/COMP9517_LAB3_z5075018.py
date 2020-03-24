"""
COMP9517 Lab 03, Week 4 -- z5075018
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from scipy import ndimage as ndi
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from sklearn.cluster import MeanShift

from PIL import Image, ImageOps, ImageEnhance

size = 100, 100

img_names = ["pics/shapes.png", "pics/car.png"]
ext_names = ["pics/aus_coins.png", "pics/kiwi.png"]

images = [i for i in img_names]
ext_images = [i for i in ext_names]


def plot_three_images(figure_title, image1, label1,
                      image2, label2, image3, label3):
    fig = plt.figure()
    fig.suptitle(figure_title)

    # Display the first image
    fig.add_subplot(1, 3, 1)
    plt.imshow(image1)
    plt.axis('off')
    plt.title(label1)

    # Display the second image
    fig.add_subplot(1, 3, 2)
    plt.imshow(image2)
    plt.axis('off')
    plt.title(label2)

    # Display the third image
    fig.add_subplot(1, 3, 3)
    plt.imshow(image3)
    plt.axis('off')
    plt.title(label3)

    #plt.show()
    fn = figure_title.split('/')[1].split('.')[0] + '_fig.png'
    plt.savefig(fn)
    print(' * Results written out to:', fn)

# added 2 args at the end (for the processed image)
def plot_four_images(figure_title, image1, label1,
                      image2, label2, image3, label3, image1_mod, label1_mod):
    fig = plt.figure()
    fig.suptitle(figure_title)
    nr = 2
    nc = 2

    # Display the first image
    fig.add_subplot(nr, nc, 1)
    plt.imshow(image1)
    plt.axis('off')
    plt.title(label1)

    fig.add_subplot(nr, nc, 2)
    plt.imshow(image1_mod, cmap=plt.get_cmap('gray'))
    plt.axis('off')
    plt.title(label1_mod)

    # Display the second image
    fig.add_subplot(nr, nc, 3)
    plt.imshow(image2)
    plt.axis('off')
    plt.title(label2)

    # Display the third image
    fig.add_subplot(nr, nc, 4)
    plt.imshow(image3)
    plt.axis('off')
    plt.title(label3)

    #plt.show()
    fn = figure_title.split('/')[1].split('.')[0] + '_fig.png'
    plt.savefig(fn)
    print(' * Results written out to:', fn)



def run_code(img_path, q2_invert=False, q2_brightval=None):
    print('Running for:', img_path)
    img = Image.open(img_path)
    img.thumbnail(size)  # Convert the image to 100 x 100
    # Convert the image to a numpy matrix
    img_mat = np.array(img)[:, :, :3]

    #
    # +--------------------+
    # |     Question 1     |
    # +--------------------+
    #
    # TODO: perform MeanShift on image
    # Follow the hints in the lab spec.

    # Step 1 - Extract the three RGB colour channels
    # Hint: It will be useful to store the shape of one of the colour
    # channels so we can reshape the flattened matrix back to this shape.
    chan_r = img_mat[:,:,0]  # (done to not interweave rgb channels w. pixels)
    chan_g = img_mat[:,:,1]
    chan_b = img_mat[:,:,2]
    chan_shape = chan_r.shape

    # Step 2 - Combine the three colour channels by flatten each channel 
	# then stacking the flattened channels together.
    # This gives the "colour_samples"
    colour_samples = np.transpose( np.array([chan_r.ravel(), chan_g.ravel(), chan_b.ravel()]) )

    # Step 3 - Perform Meanshift  clustering
    # For larger images, this may take a few minutes to compute.
    #  * Bin seeding reduces #seeds initialised; returns KMeans class
    #  * call .fit_predict() to actually run the MeanShift alg on the img
    ms_clf = MeanShift(bin_seeding=True)
    ms_labels = ms_clf.fit_predict(colour_samples)

    # Step 4 - reshape ms_labels back to the original image shape 
	# for displaying the segmentation output 
    #ms_labels = []
    ms_labels = np.reshape(ms_labels, chan_shape)
    print(" * MeanShift classes:", len(np.unique(ms_labels)))
    
    #%%
    #
    # +--------------------+
    # |     Question 2     |
    # +--------------------+
    #

    # TODO: perform Watershed on image
    # Follow the hints in the lab spec.

    # Step 1 - Convert the image to gray scale
    # and convert the image to a numpy matrix
    img_processed = img
    if q2_invert:
        print(' * Inverting image (prior to grayscale) for Q2 processing')
        img_processed = ImageOps.invert(img)
    
    img_processed = img_processed.convert('L')
    if q2_brightval: # convert to B/W image; look for white regions
        print(' * Enhancing image brightness by factor of:', q2_brightval)
        img_processed = ImageEnhance.Brightness(img_processed).enhance(q2_brightval)
    
    img_array = np.array(img_processed)

    # Step 2 - Calculate the distance transform
    # Hint: use     ndi.distance_transform_edt(img_array)
    # https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_watershed.html
    distance = ndi.distance_transform_edt(img_array)
	
    # Step 3 - Generate the watershed markers
    # Hint: use the peak_local_max() function from the skimage.feature library
    # to get the local maximum values and then convert them to markers
    # using ndi.label() -- note the markers are the 0th output to this function
    # https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_peak_local_max.html
    local_maxi = peak_local_max(distance, indices=False, labels=img_array)
    markers = ndi.label(local_maxi)[0]
	
    # Step 4 - Perform watershed and store the labels
    # Hint: use the watershed() function from the skimage.morphology library
    # with three inputs: -distance, markers and your image array as a mask
    ws_labels =  watershed(-distance, markers, mask=img_array)
    print(" * Watershed classes:", len(np.unique(ws_labels)))

    # Display the results
    plot_four_images(img_path, img, "Original", ms_labels, "MeanShift",
                     ws_labels, "Watershed", img_processed, "Watershed Preproc")

    # If you want to visualise the watershed distance markers then try
    # plotting the code below.
    # plot_three_images(img_path, img, "Original Image", -distance, "Watershed Distance",
    #                   ws_labels, "Watershed Labels")
    return img, img_processed, ms_clf, ms_labels, distance, markers, ws_labels


run_code(images[0])
run_code(images[1], True)
run_code(ext_images[0], False, 10)
run_code(ext_images[1], False, 10)

# CAR: have to invert the grayscale image for the distance image to capture the car's shape
# Watershed algorithm suggests it only works for black and white images


###########################################################
# [OLD CODE]

# for img_path in images:
#     print('Running for:', img_path)
#     img = Image.open(img_path)
#     img.thumbnail(size)  # Convert the image to 100 x 100
#     # Convert the image to a numpy matrix
#     img_mat = np.array(img)[:, :, :3]

#     #
#     # +--------------------+
#     # |     Question 1     |
#     # +--------------------+
#     #
#     # TODO: perform MeanShift on image
#     # Follow the hints in the lab spec.

#     # Step 1 - Extract the three RGB colour channels
#     # Hint: It will be useful to store the shape of one of the colour
#     # channels so we can reshape the flattened matrix back to this shape.
#     chan_r = img_mat[:,:,0]  # (done to not interweave rgb channels w. pixels)
#     chan_g = img_mat[:,:,1]
#     chan_b = img_mat[:,:,2]
#     chan_shape = chan_r.shape

#     # Step 2 - Combine the three colour channels by flatten each channel 
# 	# then stacking the flattened channels together.
#     # This gives the "colour_samples"
#     colour_samples = np.transpose( np.array([chan_r.ravel(), chan_g.ravel(), chan_b.ravel()]) )

#     # Step 3 - Perform Meanshift  clustering
#     # For larger images, this may take a few minutes to compute.
#     #  * Bin seeding reduces #seeds initialised; returns KMeans class
#     #  * call .fit_predict() to actually run the MeanShift alg on the img
#     ms_clf = MeanShift(bin_seeding=True)
#     ms_labels = ms_clf.fit_predict(colour_samples)

#     # Step 4 - reshape ms_labels back to the original image shape 
# 	# for displaying the segmentation output 
#     #ms_labels = []
#     ms_labels = np.reshape(ms_labels, chan_shape)
#     print(" * MeanShift classes:", np.unique(ms_labels))


#     #%%
#     #
#     # +--------------------+
#     # |     Question 2     |
#     # +--------------------+
#     #

#     # TODO: perform Watershed on image
#     # Follow the hints in the lab spec.

#     # Step 1 - Convert the image to gray scale
#     # and convert the image to a numpy matrix
#     img_array = np.array(img.convert('L'))
	
#     # Step 2 - Calculate the distance transform
#     # Hint: use     ndi.distance_transform_edt(img_array)
#     # https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_watershed.html
#     distance = ndi.distance_transform_edt(img_array)
	
#     # Step 3 - Generate the watershed markers
#     # Hint: use the peak_local_max() function from the skimage.feature library
#     # to get the local maximum values and then convert them to markers
#     # using ndi.label() -- note the markers are the 0th output to this function
#     # https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_peak_local_max.html
#     local_maxi = peak_local_max(distance, indices=False, labels=img_array)
#     markers = ndi.label(local_maxi)[0]
	
#     # Step 4 - Perform watershed and store the labels
#     # Hint: use the watershed() function from the skimage.morphology library
#     # with three inputs: -distance, markers and your image array as a mask
#     ws_labels =  watershed(-distance, markers, mask=img_array)
#     print(" * Watershed classes:", np.unique(ws_labels))

#     # Display the results
#     plot_three_images(img_path, img, "Original Image", ms_labels, "MeanShift Labels",
#                       ws_labels, "Watershed Labels")

#     # If you want to visualise the watershed distance markers then try
#     # plotting the code below.
#     # plot_three_images(img_path, img, "Original Image", -distance, "Watershed Distance",
#     #                   ws_labels, "Watershed Labels")

#%%
#
# +-------------------+
# |     Extension     |
# +-------------------+
#
# Loop for the extension component
# for img_path in ext_images:
#     img = Image.open(img_path)
#     img.thumbnail(size)


#     # TODO: perform meanshift on image
#     ms_labels = img  # CHANGE THIS

#     # TODO: perform an optimisation (pre-processing) and then watershed on image
#     ws_labels = img  # CHANGE THIS

#     # Display the results
#     plot_three_images(img_path, img, "Original Image", ms_labels, "MeanShift Labels",
#                       ws_labels, "Watershed Labels")

