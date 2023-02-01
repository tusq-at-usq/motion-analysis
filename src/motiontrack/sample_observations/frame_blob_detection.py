""" Code to get blobs from individual frame
"""

from PIL import Image
import numpy as np
import cv2
import pandas as pd

TOP_IMAGE_EXAMPLE = "./image_data/top/example.tif"
EAST_IMAGE_EXAMPLE = "./image_data/east/example.tif"

def get_blobs(im):

    thr_max = 255
    block_size = 21
    scale = 11

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
    im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
     
    # Show keypoints
    points = np.array([k.pt for k in keypoints])
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
    points = get_blobs(np.array(Image.open(EAST_IMAGE_EXAMPLE)))


