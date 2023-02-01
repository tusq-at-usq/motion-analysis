import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import pickle

file = './drop_4/top_cam/top_cam_C002H001S0001000580.tif'
with open('./top_cam_single/saved.pkl', 'rb') as f:
    ret, mtx, dist, rvecs, tvecs, _ = pickle.load(f)

def plot(im,line_x, line_y):
    plt.figure(figsize=(12,12))
    plt.imshow(im)
    plt.plot(line_x.T, line_y.T, marker='x')
    plt.show(block=False)
    plt.pause(0.1)
    input("Press to continue")
    plt.close()

img = cv.imread(file)
img = cv.bitwise_not(img)
blur = cv.GaussianBlur(gray, (5,5), 0)
thresh = cv.adaptiveThreshold(blur,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.ADAPTIVE_THRESH_GAUSSIAN_C,151,3)
contours, heirachy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

plot(thresh, np.array([]), np.array([]))
plot(img, np.array([]), np.array([]))
