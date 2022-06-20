# Instructions for the *cv2blobdetect* module

The *cv2blobdetect* module has been designed with three separate classes:

1. Tracker
2. Locater
3. BlobHandler

The Tracker is a parent class that was designed to be a container for the different stages of the tracking process, by binding a reference to the parent in each child class at instantiation. At the moment there are only two such child classes, but the original development also contained a silhouette fitter which necessitated the use of a parent object. At object instantiation, you only need to provide the path to the video to be analysed.

The Locater object is where most of the work is done and where you do most of your project-specific tweaking. It applies the filtering operations to each frame of the input video, runs blob detection and saves a list of the keypoints (blobs) for each one. We'll talk about how to do that in the example below.

The final class is the BlobHandler. This takes the list of keypoints found by the Locater and converts them into a preferred export format which is then saved as a CSV file, and is the output that gets piped to the Kalman filtering module.

# Example of Usage.

trk = Tracker(vid='pathtovideo.mp4')

loc = trk.locater

----------The next two lines are optional--------

loc.frame_start = 31

loc.frames_to_show = 100

loc.locate()

trk.blob_handler.convert()

trk.blob_handler.export('filename')

# Tips:

1. Most of the tweaking you will need to do is in:

Locater.preprocess_filter()

Locater.find_contours()

Locater.detect_blobs()

2. You may also find it beneficial to play around with the settings of Locater.blob_params

# To do:

1. The Tracker and Locater objects could probably be merged into a single class. As mentioned, it was originally designed with some other components that no longer exist, but I haven't bothered to streamline that yet.
1. The Locater could save some time by just saving the keypoints in the export format from the get-go. This module and the Kalman filter blob tracking/pose estimation code were developed separately and thus adopted their own conventions, so the BlobHandler's only real function is to translate between them.
1. Error handling - there isn't much at the moment.