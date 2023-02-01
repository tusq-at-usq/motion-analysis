""" Code to get blobs from individual frame
"""

from PIL import Image
import numpy as np
import cv2
import pandas as pd

#  TOP_IMAGE_EXAMPLE = "./image_data/run1329_top/run1329_C002H001S0001000200.tif"
TOP_IMAGE_EXAMPLE = "./image_data/run1329_east/run1329_C001H001S0001000450.tif"
EAST_IMAGE_EXAMPLE = "./image_data/run1329_east/run1329_C001H001S0001000900.tif"

def get_blobs(im):

    #  im = (im/8).astype('uint8')

    #  im = cv2.GaussianBlur(im,(5,5),0)

    thr_max = 255
    block_size = 21
    scale = 11

    #  im = cv2.resize(im,im.shape[::-1])
    #  im = cv2.bilateralFilter(im, 5,75,75)
    #  im = cv2.bilateralFilter(im, 5,75,75)

    #  im = cv2.medianBlur(im, 3)
    #  im = cv2.gaussianBlur(im, 3)
    #  im = cv2.GaussianBlur(im,(5,5),0)
    #  im = cv2.GaussianBlur(im,(3,3),0)

    #  im = cv2.adaptiveThreshold(im, thr_max, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                #  cv2.THRESH_BINARY, block_size, scale)

    #  im = cv2.adaptiveThreshold(im, thr_max, cv2.ADAPTIVE_THRESH_MEAN_C,\
                #  cv2.THRESH_BINARY, block_size, scale)

    #  res, im = cv2.threshold(im,0,255,cv2.THRESH_OTSU)

    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()
     
    # Change thresholds
    params.minThreshold = 1;
    params.maxThreshold = 10000
     
    # Filter by Area.
    params.filterByArea = True
    params.minArea = 100
    params.maxArea=500
     
    # Filter by Circularity

    params.filterByCircularity = True
    params.minCircularity = 0.5
     
    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.3
     
    # Filter by Inertia
    params.filterByInertia = False
    params.minInertiaRatio = 0.03
    #  params.maxInertiaRatio = 0.2
     
    # Set up the detector with default parameters.
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs.
    keypoints = detector.detect(im)
     
    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
     
    # Show keypoints
    #  cv2.imshow("Keypoints", im_with_keypoints)
    #  cv2.waitKey(0)

    points = np.array([k.pt for k in keypoints])
    #  points[:,1] = im.shape[0]-points[:,1]
    sizes = np.array([k.size for k in keypoints])**0.5 /  np.pi
    return points, sizes

def get_blobs_from_csv(fpath,frame_num):

    df = pd.read_csv(fpath,index_col=0)

    df_slice = df[df['FrameID']==frame_num]

    xy = df_slice[['x','y']].values       

    sizes = df_slice[['Size']].values       

    return xy, sizes

if __name__ == "__main__":
    points = get_blobs(np.array(Image.open(TOP_IMAGE_EXAMPLE)))


