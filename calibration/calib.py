import numpy as np
import cv2
import glob
#Function that Downsamples image x number (reduce_factor) of times.
def downsample_image(image, reduce_factor):
	for i in range(0,reduce_factor):
		#Check if image is color or grayscale
		if len(image.shape) > 2:
			row,col = image.shape[:2]
		else:
			row,col = image.shape

		image = cv2.pyrDown(image, dstsize= (col//2, row // 2))
	return image

chessboard_size = (7,9)
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((np.prod(chessboard_size),3),dtype=np.float32)
objp[:,:2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1,2)

debug_show = True 
# Arrays to store object points and image points from all the images.
objpointsL = [] # 3d point in real world space
imgpointsL = [] # 2d points in image plane.
objpointsR = []
imgpointsR = []

images = sorted(glob.glob('chess/left/*.jpg'))

for fname in images:
    img = cv2.imread(fname)
    grayL = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret,corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpointsL.append(objp)

        cornersL = cv2.cornerSubPix(grayL,cornersL,(11,11),(-1,-1),criteria)
        imgpointsL.append(cornersL)
        if debug_show:
            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, chessboard_size, corners,ret)
            cv2.imshow('img',img)
            cv2.waitKey(1000)

retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpointsL, imgpointsL, grayL.shape[::-1], None, None)
#print(retL, mtxL, distL, rvecsL, tvecsL)

#Right view calibration
images = sorted(glob.glob('chess/right/*.jpg'))

for fname in images:
    img = cv2.imread(fname)
    grayR = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, cornersR = cv2.findChessboardCorners(grayR, (6,9),None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpointsR.append(objp)

        cornersR = cv2.cornerSubPix(grayR,cornersR,(11,11),(-1,-1),criteria)
        imgpointsR.append(cornersR)

        if debug_show:
            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (6,9), cornersR,ret)
            cv2.imshow('img',img)
            cv2.waitKey(1000)

retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpointsR, imgpointsR, grayR.shape[::-1], None, None)
#print(retR, mtxR, distR, rvecsR, tvecsR)


#Undistort test
images = sorted(glob.glob('left/*.jpg'))

for fname in images:
    img = cv2.imread(fname)
    h,  w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtxL, distL, (w,h), 1, (w,h))

    # undistort
    dst = cv2.undistort(img, mtxL, distL, None, newcameramtx)
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    cv2.imshow('original',img)
    cv2.imshow('undistorted',dst)
    cv2.waitKey(1000)
    raw_input('')


#retval,cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(objpointsL, imgpointsL, imgpointsR, (320,240))
cv2.destroyAllWindows()
