import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Read in the saved camera matrix and distortion coefficients
# These are the arrays you calculated using cv2.calibrateCamera()
dist_pickle = pickle.load( open( "wide_dist_pickle.p", "rb" ) )
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

# Read in an image
img = cv2.imread('test_image2.png')
nx = 8 # the number of inside corners in x
ny = 6 # the number of inside corners in y

# MODIFY THIS FUNCTION TO GENERATE OUTPUT 
# THAT LOOKS LIKE THE IMAGE ABOVE
def corners_unwarp(img, nx, ny, mtx, dist):
    
    # Do camera calibration given object points and image points
    img_size = (img.shape[1], img.shape[0])
    # Do image undistort
    undistort = cv2.undistort(img, mtx, dist, None, mtx)
    gray = cv2.cvtColor(undistort, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)
    # If found, add object points, image points
    if ret == True:
        # Draw and display the corners
        
        cv2.drawChessboardCorners(undistort, (nx,ny), corners, ret)
        """    
        cv2.circle(undistort,(corners[0][0][0],corners[0][0][1]), 63, (0,0,255), -1)
        cv2.circle(undistort,(corners[47][0][0],corners[47][0][1]), 63, (0,0,255), -1)
        cv2.circle(undistort,(corners[7][0][0],corners[7][0][1]), 63, (0,0,255), -1)
        cv2.circle(undistort,(corners[40][0][0],corners[40][0][1]), 63, (0,0,255), -1)
        """       
       # src = np.float32([[corners[0][0],corners[7][0],[corners[47][0],corners[40][0])
        
        src = np.float32([[corners[0][0][0],corners[0][0][1]],[corners[7][0][0],corners[7][0][1]],[corners[47][0][0],corners[47][0][1]],[corners[40][0][0],corners[40][0][1]]])
        dst = np.float32([[20,20],[img.shape[1]-20,20],[img.shape[1]-20,img.shape[0]-20],[20,img.shape[0]-20]])
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(undistort, M, img_size, flags=cv2.INTER_LINEAR)
       
    

    # Pass in your image into this function
    # Write code to do the following steps
    # 1) Undistort using mtx and dist
    # 2) Convert to grayscale
    # 3) Find the chessboard corners
    # 4) If corners found: 
            # a) draw corners
            # b) define 4 source points src = np.float32([[,],[,],[,],[,]])
                 #Note: you could pick any four of the detected corners 
                 # as long as those four corners define a rectangle
                 #One especially smart way to do this would be to use four well-chosen
                 # corners that were automatically detected during the undistortion steps
                 #We recommend using the automatic detection of corners in your code
            # c) define 4 destination points dst = np.float32([[,],[,],[,],[,]])
            # d) use cv2.getPerspectiveTransform() to get M, the transform matrix
            # e) use cv2.warpPerspective() to warp your image to a top-down view
    #delete the next two lines
    #M = None
    #warped = np.copy(undistort) 
    return warped, M

top_down, perspective_M = corners_unwarp(img, nx, ny, mtx, dist)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 18))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(top_down)
ax2.set_title('Undistorted and Warped Image', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
