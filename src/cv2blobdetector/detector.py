import cv2, numpy as np
import pandas as pd

class Tracker:
    
# =============================================================================
#     Parent class which combines the object locater, blob handler and silhouette
#     fitter, though the silhouette fitter won't be much further developed.
#     
#     Usage:
#         
#         1. Instantiate the Tracker object with the only argument being the path
#             to the video file.
#         2. The Tracker creates a Locater object whose job is to process the
#             image and find blobs.
#         3. The Tracker object also creates a BlobHandler. which is then used
#            to convert the blobs found by the Locater into the export format.
#             
#     Example:
#         
#         trk = Tracker(vid='pathtovideo.mp4')
#         loc = trk.locater
#
#         ----------The next two lines are optional--------
# 
#         loc.frame_start = 31
#         loc.frames_to_show = 100
# 
#         loc.locate()
#         trk.blob_handler.convert()
#         trk.blob_handler.export('filename')
# 
#         Tips:
#            1. Most of the tweaking you will need to do is in:
#               Locater.preprocess_filter()
#               Locater.find_contours()
#               Locater.detect_blobs()
#            2. You may also find it beneficial to play around with the settings
#                of Locater.blob_params
# 
# =============================================================================
    
    def __init__(self,vid=None):
            
        self.video_file_name = vid
        
        #Load video into OpenCV's videoReader
        self.set_data()
            
        #Bind the object detector
        self.locater = Locater(self)
        
        #Bind the Blob handler
        self.blob_handler = BlobHandler(self)
                
    def set_data(self):
        
        """
        Set the video capture object
        """
        
        f = self.video_file_name
        if f:
            self.cap = cv2.VideoCapture(f)    
        else:
            self.cap = None
            
    def locate_object(self):
        
        self.locater.locate()
        
class Locater:
    
    """
    Object locater class
    """
    
    def __init__(self,parent,kernel=1e-2*np.ones((10,10)),
                 frame_start=None,frames_to_show=None,downscale_factor=1,render=True,background=0,
                 only_circular_blobs=False,only_convex_blobs=False,opening_kernel=np.ones((100,100)),
                 closing_kernel=np.ones((15,15))):
                
        # Bind the parent tracker so the video capture can be accessed
        self.parent = parent
        
        #Convolution kernel
        self.kernel = kernel
        self.opening_kernel = opening_kernel
        self.closing_kernel = closing_kernel
        
        #Frame number to start at
        self.frame_start = frame_start
        
        #Number of frames to analyse in the locater
        self.frames_to_show = frames_to_show
        
        self.outframes = []
        
        self.img_shape = None
        
        #Boolean whether to show the detector output in matplotlib before image rendering
        self.render = render
        
        #Set up the blob detector using known working parameters
        self.blob_params = cv2.SimpleBlobDetector_Params()
        self.blob_params.minArea = 20
        self.blob_params.filterByConvexity = only_convex_blobs
        self.blob_params.filterByCircularity = only_circular_blobs
        self.blob_detector = cv2.SimpleBlobDetector_create(self.blob_params)
        
        #Bounding box dimensions can be analysed with FFT for spin rates
        #Only set up to track one object atm
        self.bbox_dims = []
        
        # integer: light background -> 0; dark background -> 1
        self.background = background
                
        self.blobs = []
        
    def locate(self):
        
        try:
            assert self.parent.cap is not None
        except AssertionError:
            print('The video reader has not been set. Terminating.')
            return False
        
        #Reload the video to start
        self.parent.set_data()

        #Container for the frame count and location of object detected in each frame
        self.tracks = []
                
        self.frame_counter = -1

        while True:
                                                    
            self.frame_counter +=1 
            
            ret, frame = self.parent.cap.read()
            
            #Check start/stop conditions
            if self.frame_start and self.frame_counter < self.frame_start:
                continue
            if self.frames_to_show:
                start = self.frame_start if self.frame_start else 0
                finish = start + self.frames_to_show
                if self.frame_counter > finish: break
                    
            #Break if no image read
            if not ret:
                print('End of file.')
                break
            
            #Save original images in colour and grayscale
            self.cur_img = frame
            self.cur_img_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            
            #Containers for detected features to be added later
            self.current_frame_tracks = []
            self.current_frame_contours = []
            
            # =============================
            #  Start filtering operations
            # =============================
            
            #Initial thresholding
            self.preprocess_filter()
            #Contour finding - the contour is needed to fill a polygon
            self.find_contours()
            #Use the object mask to recover section of the original image
            self.isolate_masked_section()
            #Detect and draw blobs inside that masked section
            self.detect_blobs()
                
            # =============================
            #  End filtering operations
            # =============================
            
            #Invert back to original colours (dark objects in light BG)
            self.cur_img_object_masked = np.invert(self.cur_img_object_masked)
            
            #Put text and annotations on the video
            self.annotate()
            
            self.outframes.append(np.hstack([self.cur_img_grayframe_downsized,self.cur_img_object_masked_downsized]))
            
            cv2.imshow('Original Video',self.cur_img_grayframe_downsized)
            cv2.imshow('Filtered Objects',self.cur_img_object_masked_downsized)
            cv2.imshow('Filtered Blobs',self.blob_img)
            cv2.waitKey(20);

            self.tracks.append(self.current_frame_tracks)
                            
        cv2.destroyAllWindows()
        
        #Create the output video
        if self.render:
            self.render_output()

    def preprocess_filter(self):
        
        # =============================================
        # Process the image prior to contour detection
        # =============================================
        
        #Create inverted mask (high intensity objects on light BG)
        if self.background == 1:
            mask = np.invert(self.cur_img_gray)
        else:
            mask = self.cur_img_gray.copy()
        
        self.grayframe = mask.copy()
  
        #Perform thresholding operations
        _, mask = cv2.threshold(mask,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        mask = cv2.medianBlur(mask,11)
                    
        self.preprocessed_image = mask.copy()
        
    def find_contours(self):
        
        mask = self.preprocessed_image
        
        #Find contours
        cnts, _ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
        #Loop through contours, ignoring the ones near corners
        self.draw_contours = []
        for c in cnts:
            dist_to_top_left = cv2.pointPolygonTest(c,(0,0), True)
            dist_to_top_right = cv2.pointPolygonTest(c,(mask.shape[1],0),True)
            dist_to_bottom_left = cv2.pointPolygonTest(c,(0,mask.shape[0]),True)
            dist_to_bottom_right = cv2.pointPolygonTest(c,(mask.shape[1],mask.shape[0]),True)
            
            if any([abs(x) < 10 
                    for x in 
                    [dist_to_top_left, dist_to_top_right, dist_to_bottom_left, dist_to_bottom_right]]):
                continue
            
            #Leave contours that are too small to be the object
            if cv2.contourArea(c) < 1e3:
                continue
            if cv2.contourArea(c) > 1e5:
                continue
                                            
            M = cv2.moments(c)
            cy = int(M['m10']/M['m00'])
            cx = int(M['m01']/M['m00'])
            self.cy, self.cx = cy, cx
            
            c = cv2.convexHull(c)
            
            self.draw_contours.append(c)
            self.current_frame_tracks.append([self.frame_counter,cx,cy])
            self.current_frame_contours.append(c)
            
            #Draw circles at the polygon COMs
            try:
                cv2.circle(self.grayframe,(cy,cx), 20, 0,-1)    
            except UnboundLocalError:
                pass
            
    def isolate_masked_section(self):
        
        #Create a blank template for drawing on
        template = np.zeros_like(self.cur_img_gray)
        
        #Draw a filled-in polygon of the object contour
        cv2.fillPoly(template,self.draw_contours,255)
        
        #This template can now be used as a filled polygon mask for 
        #detected objects
        self.cur_img_isolated_binary_objs = template.copy() #dark cube
        self.cur_img_isolated_binary_objs_inv = np.invert(self.cur_img_isolated_binary_objs) #bright cube
        
        #Bitwise-and operation creates a mask of the original image
        #retaining only the section with the detected object/s
        self.cur_img_object_masked = cv2.bitwise_and(self.grayframe,template)
        
    def annotate(self):
            
            #Create a colour copy of the filtered image
            #Can be used to overlay coloured markings
            self.cur_img_isolated_binary_objs_color = cv2.cvtColor(self.cur_img_object_masked,cv2.COLOR_GRAY2BGR)
            
            grayframe = self.grayframe
            cur_img_object_masked = self.cur_img_object_masked
            #Text labels
            cv2.putText(grayframe,'Original Video',(150,130),cv2.FONT_HERSHEY_SIMPLEX,3,0,5)
            cv2.putText(grayframe,f'Center at ({self.cy},{self.cx})',(150,200),cv2.FONT_HERSHEY_SIMPLEX,1,0,5)
            cv2.putText(grayframe,f'Frame: {self.frame_counter}',(150,260),cv2.FONT_HERSHEY_SIMPLEX,1,0,5)
            cv2.putText(cur_img_object_masked,'Object Isolated',(150,130),cv2.FONT_HERSHEY_SIMPLEX,3,0,5)
            cv2.putText(cur_img_object_masked,f'{len(self.current_frame_blob_keypoints)} blobs detected',(400,200),cv2.FONT_HERSHEY_SIMPLEX,1,0,5)

            #Downsize images prior to showing
            self.cur_img_grayframe_downsized = cv2.resize(grayframe,(500,500))
            self.cur_img_object_masked_downsized = cv2.resize(cur_img_object_masked,(500,500))
            self.blob_img = cv2.resize(self.blob_img,(500,500))
            cv2.putText(self.blob_img,'Filtered - Blobs Only',(25,50),cv2.FONT_HERSHEY_SIMPLEX,1,0,2)
        
    def detect_blobs(self):
        
        pad = 0 # pix
        
        self.current_frame_blob_keypoints = []
        
        self.blob_image = np.zeros_like(self.cur_img)

        if not self.current_frame_contours: return []

        for cnt in self.current_frame_contours:
                                    
            """
            CAUTION: OpenCV uses unusual image coordinate axes.
            
            0: ---->
            1: |
               | 
               |
              \/
              
            Images matrix axes are
            
            1: ---->
            0: |
               | 
               |
              \/
            
            """
            
            #get bounding box for the object
            x,y,w,h = cv2.boundingRect(cnt)
            self.bbox_dims.append([w,h])
            
            #Get local coordinates of the contour
            local_cnt = cnt.copy()
            local_cnt[:,:,0] -= x-pad
            local_cnt[:,:,1] -= y-pad
            
            self.local_cnt = local_cnt

            #Isolate region of interest
            iso = self.cur_img_gray.copy()
            
            #Cut the image down to only the area defined by the contour
            iso = iso[y-pad:y+h+pad,x-pad:x+w+pad]
            
            roi_mask = self.cur_img_isolated_binary_objs[y-pad:y+h+pad,x-pad:x+w+pad]
            
            roi_mask = cv2.erode(roi_mask,np.ones((5,5)))
            self.roi_mask = roi_mask
            
            #Increase contrast of image
            _,thresh = cv2.threshold(iso,100,255,cv2.THRESH_TOZERO_INV+cv2.THRESH_BINARY)
            
            #Remove any data outside of the contour internal area
            iso[roi_mask==0] = 255

            #Draw over the contour so it isn't visible
            cv2.drawContours(iso,[local_cnt],-1,255,30)
            
            #Detect blobs
            kp = self.blob_detector.detect(iso)
            
            #Change the blob locations into global CS
            for pt in kp:
                 
                blob_x,blob_y = pt.pt
                #But also save the floating point versions for subpixel accuracy
                pt.pt = (x+blob_x-pad,y+blob_y-pad)
                self.parent.blob_handler.blobs.append([self.frame_counter,pt.pt,pt.size,pt.angle])
                pt.pt = [int(x) for x in pt.pt]
                
            self.current_frame_blob_keypoints.extend(kp)
        
        #Draw the blobs on the masked image
        for blob in self.current_frame_blob_keypoints:
            center = ([int(x) for x in blob.pt])
            cv2.circle(self.cur_img_object_masked,center,3,0,-1)
            
        self.blob_img = iso
        
        return self.current_frame_blob_keypoints
           
    def get_image_shape(self):
        if self.cur_img is not None:
            self.img_shape = self.cur_img.shape[:2]
        else:
            print('Error getting image shape')
            
        return self.img_shape
    
    def render_output(self):
        
        outframes = [cv2.cvtColor(frame,cv2.COLOR_GRAY2BGR) for frame in self.outframes]
        
        if not self.outframes:
            print('Error - no frames to render.')
            return False
        
        vname = self.parent.video_file_name
        
        #Video writer requires the 2D image shape
        shape = (outframes[0].shape[1],outframes[0].shape[0])
        
        out = cv2.VideoWriter(f'analysis_of_{vname}'.replace('mp4','avi'),cv2.VideoWriter_fourcc(*'DIVX'),30,shape)

        for i in range(len(self.outframes)):
            out.write(outframes[i])
        
        out.release()   
    
class BlobHandler:
    
    def __init__(self,parent):
        
        self.blobs = []
        self.parent = parent
        
    def convert(self,method='cartesian'):
        
        #Convert XY coordinates from image space into cartesian frame
        
        shape = self.parent.locater.cur_img_gray.shape
        
        for i, r in enumerate(self.blobs):
            
            x,y = r[1]
            
            if method == 'cartesian':
                y = shape[0]-y
                
            self.blobs[i][1] = (x,y)
                    
    def export(self,fname):
        
        out_lines = []
        
        for r in self.blobs:
            
            c,(x,y),size,angle = r
            
            out_lines.append([c,0,x,y,size,angle]) 
            
        out_lines = np.vstack(out_lines)
        
        self.df = pd.DataFrame(data=out_lines,columns=['FrameID','Region','x','y','Size','Angle'])
        self.df.FrameID = self.df.FrameID.astype(int)
        self.df.Region = self.df.Region.astype(int)
        self.df.to_csv(f'{fname}.csv')
    
if __name__ == '__main__':
    trk = Tracker(vid='pathtovideo.mp4')
    loc = trk.locater
    loc.frame_start = 31
    # trk.locater.frames_to_show = 100
    loc.locate()
    trk.blob_handler.convert()
    trk.blob_handler.export('somevideo')

    
