import cv2
import numpy as np
import skimage.color as color

def instance_watershed(segmented, threshold):
    segmented = (segmented[0,:,:,:] > threshold).astype(np.uint8)
    segmented2 = cv2.colorChange(segmented,cv2.COLOR_BAYER_BG2GRAY)
    img_grey = segmented2[:,:,0]


    ret1, thresh = cv2.threshold(img_grey, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations = 2)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2,5)
    sure_bg = cv2.dilate(opening, kernel,iterations=10)


    ret2, sure_fg = cv2.threshold(dist_transform, 0.2*dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    unknown = cv2.subtract(sure_bg, sure_fg)

    ret3, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 10

    markers[unknown==255] = 0

    markers =  cv2.watershed(segmented, markers)

    segmented[markers == -1] = [0,255,255]

    img2 = color.label2rgb(markers, bg_label=0)
    return img2