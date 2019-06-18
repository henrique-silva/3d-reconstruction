import numpy as np
import cv2
import glob
import PIL.Image
import PIL.ExifTags

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
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = sorted(glob.glob('chess/left/*.jpg'))

for fname in images:
    print('Calibrating with image '+fname)
    img = cv2.imread(fname)
    img = downsample_image(img, 2)

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray',gray)
    cv2.waitKey(10)
    # Find the chess board corners
    ret,corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners)
        if debug_show:
            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, chessboard_size, corners,ret)
            cv2.imshow('img',img)
            cv2.waitKey(10)
    else:
        print('Could not find points in image!')
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

#Get exif data in order to get focal length.
exif_img = PIL.Image.open(images[0])

exif_data = {
	PIL.ExifTags.TAGS[k]:v
	for k, v in exif_img._getexif().items()
	if k in PIL.ExifTags.TAGS}

#Get focal length in tuple form
focal_length_exif = exif_data['FocalLength']

#Get focal length in decimal form
focal_length = float(focal_length_exif[0])/focal_length_exif[1]





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

cv2.destroyAllWindows()
