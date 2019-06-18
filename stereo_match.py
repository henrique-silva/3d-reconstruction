import cv2
import numpy as np
import argparse

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

#Function to create point cloud file
def create_output(vertices, colors, filename):
	colors = colors.reshape(-1,3)
	vertices = np.hstack([vertices.reshape(-1,3),colors])

	ply_header = '''ply
		format ascii 1.0
		element vertex %(vert_num)d
		property float x
		property float y
		property float z
		property uchar red
		property uchar green
		property uchar blue
		end_header
		'''
	with open(filename, 'w') as f:
		f.write(ply_header %dict(vert_num=len(vertices)))
		np.savetxt(f,vertices,'%f %f %f %d %d %d')

parser = argparse.ArgumentParser()
parser.add_argument('imgL', type=str, help='Path to Left image')
parser.add_argument('imgR', type=str, help='Path to Right image')
parser.add_argument('-o','--output', type=str, default='reconstructed.ply', help='Output 3d file path')
parser.add_argument('-p','--parameters', type=str, default='./camera_params/', help='Camera parameters folder')
parser.add_argument('-m','--matcher', choices=['Normal','SemiGlobal'], default='SemiGlobal', help='Stereo Matcher algorithm')
parser.add_argument('-w','--window_size', type=int, default=5, help='SemiGlobal Stereo Matcher window size')
parser.add_argument('-d','--downsample', type=int, default=1, help='Downsample factor')
parser.add_argument('--debug', action='store_true', help='Show image processing mid-steps')
parser.add_argument('--no_undistort', action='store_true', help='Don\'t attempt to undistort images')
args = parser.parse_args()

imgL = cv2.imread(args.imgL)
imgR = cv2.imread(args.imgR)

if not args.no_undistort:
	#Load camera parameters
	ret_val = np.load(args.parameters+'RetVal.npy')
	camera_matrix = np.load(args.parameters+'CameraMatrix.npy')
	distortion_coeffs = np.load(args.parameters+'DistortionCoeffs.npy')

	#Get height and width. (Both pictures must have the same size)
	height, width = imgL.shape[:2]

	#Get optimal camera matrix for better undistortion
	new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_coeffs,(width, height), 1, (width, height))

	#Undistort images
	imgL_undistorted = cv2.undistort(imgL, camera_matrix, distortion_coeffs, None, new_camera_matrix)
	imgR_undistorted = cv2.undistort(imgR, camera_matrix, distortion_coeffs, None, new_camera_matrix)

	#Downsample each image if asked
	imgL_downsampled = downsample_image(imgL_undistorted,args.downsample)
	imgR_downsampled = downsample_image(imgR_undistorted,args.downsample)

	grayL = cv2.cvtColor(imgL_downsampled, cv2.COLOR_BGR2GRAY)
	grayR = cv2.cvtColor(imgR_downsampled, cv2.COLOR_BGR2GRAY)

else:
	#Downsample each image if asked
	imgL_downsampled = downsample_image(imgL,args.downsample)
	imgR_downsampled = downsample_image(imgR,args.downsample)

	grayL = cv2.cvtColor(imgL_downsampled, cv2.COLOR_BGR2GRAY)
	grayR = cv2.cvtColor(imgR_downsampled, cv2.COLOR_BGR2GRAY)

if (args.debug):
	cv2.imshow('Right image', grayR)
	cv2.waitKey(500)
	cv2.imshow('Left image', grayL)
	cv2.waitKey(500)

if (args.matcher == 'Normal'):
	stereoMatcher = cv2.StereoBM_create()
	depth = stereoMatcher.compute(grayL, grayR)
	if (args.debug):
		depthNorm = cv2.normalize(src=depth, dst=None, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
		depthNorm = np.uint8(depthNorm)
		cv2.imshow('depth', depthNorm)

elif (args.matcher == 'SemiGlobal'):
	# wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
	stereoMatcherSG = cv2.StereoSGBM_create(
    	minDisparity=0,
		numDisparities=128,             # max_disp has to be dividable by 16 f. E. HH 192, 256
    	blockSize=5,
    	P1=8 * 3 * args.window_size ** 2,    # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
    	P2=32 * 3 * args.window_size ** 2,
    	disp12MaxDiff=1,
    	uniquenessRatio=15,
    	speckleWindowSize=0,
    	speckleRange=2,
    	preFilterCap=63,
    	mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
	)
	depthSG = stereoMatcherSG.compute(grayL, grayR)
	if (args.debug):
		depthNormSG = cv2.normalize(src=depthSG, dst=None, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
		depthNormSG = np.uint8(depthNormSG)
		cv2.imshow('depthSG', depthNormSG)

if (args.debug):
	cv2.waitKey()

#######################################################################
##### 3D reprojection
######################################################################

h,w = grayL.shape[:2]

#Load focal length.
focal_length = np.load(args.parameters+'FocalLength.npy')

#Perspective transformation matrix
#This transformation matrix is derived from Prof. Didier Stricker's power point presentation on computer vision.
#Link : https://ags.cs.uni-kl.de/fileadmin/inf_ags/3dcv-ws14-15/3DCV_lec01_camera.pdf
Q = np.float32([[1,0,0,0],
				[0,-1,0,0],
				[0,0,focal_length*0.05,0], #Focal length multiplication obtained experimentally.
				[0,0,0,1]])

#Reproject points into 3D
points_3D = cv2.reprojectImageTo3D(depthSG, Q)

#Get color points
colors = cv2.cvtColor(imgL_downsampled, cv2.COLOR_BGR2RGB)

#Get rid of points with value 0 (i.e no depth)
mask_map = depthSG > depthSG.min()

#Mask colors and points.
output_points = points_3D[mask_map]
output_colors = colors[mask_map]

#Define name for output file
#Generate point cloud
print ("\n Creating the output file... \n")
create_output(output_points, output_colors, args.output)
