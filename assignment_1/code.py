#!/usr/env/bin python3

import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
import os

from utils import *
from functions import *

'''
Starter code for loading files, calibration data, and transformations
'''

# File paths
calib_dir = os.path.abspath('./data/calib')
image_dir = os.path.abspath('./data/image')
lidar_dir = os.path.abspath('./data/velodyne')
sample = '000000'

# Load the image
image_path = os.path.join(image_dir, sample + '.png')
image = img.imread(image_path)

# Load the LiDAR points
lidar_path = os.path.join(lidar_dir, sample + '.bin')
lidar_points = load_velo_points(lidar_path)
print_func(lidar_points, "lidar_points")

# Load the body to camera and body to LiDAR transforms
body_to_lidar_calib_path = os.path.join(calib_dir, 'calib_imu_to_velo.txt')
T_lidar_body = load_calib_rigid(body_to_lidar_calib_path)
# print_func(T_lidar_body, "T_lidar_body")
T_body_lidar = np.linalg.inv(T_lidar_body)

# Load the camera calibration data
# Remember that when using the calibration data, there are 4 cameras with IDs
# 0 to 3. We will only consider images from camera 2.
lidar_to_cam_calib_path = os.path.join(calib_dir, 'calib_velo_to_cam.txt')
cam_to_cam_calib_path = os.path.join(calib_dir, 'calib_cam_to_cam.txt')
cam_calib = load_calib_cam_to_cam(lidar_to_cam_calib_path, cam_to_cam_calib_path)
intrinsics = cam_calib['K_cam2']
T_cam2_lidar = cam_calib['T_cam2_velo']
print_func(intrinsics, "intrinsics")
print_func(T_cam2_lidar, "T_cam2_lidar")

'''
For you to complete:
Part 1: Convert LiDAR points from LiDAR to body frame (for depths)
Note that the LiDAR data is in the format (x, y, z, r) where x, y, and z are
distances in metres and r is a reflectance value for the point which can be
ignored. x is forward, y is left, and z is up. Depth can be calculated using
# d^2 = x^2 + y^2 + z^2
'''
# 1) LiDAR Feature Coordinate Transformation

# Answer 1.1
C_BL = rotation_matrix_tait_bryan(-10, -23, -90)
T_BL = transformation_matrix(C_BL, 2.57, -0.52, 1.32)
print_func(C_BL, "C_BL")
print_func(T_BL, "T_BL")

# Answer 1.2
pose_lidar_test_1 = np.array([[3.64], [8.30], [2.45]])
pose_body_test_1 = transform_point(T_BL, pose_lidar_test_1)
print_func(pose_body_test_1, "pose_body_test_1")

# Answer 1.3
T_LB = np.around(np.linalg.inv(T_BL), 3)
print_func(T_LB, "T_LB")

# Convert all LiDAR points from LiDAR to body frame
pose_lidar = np.transpose(lidar_points)[0:3, :]
print_func(pose_lidar, "pose_lidar")
pose_body = transform_point(T_body_lidar, pose_lidar)
print_func(pose_body, "pose_body")

# Calculate depth of lidar points in body frame
depth_lidar = calculate_depth(pose_body)
print_func(depth_lidar, "depth_lidar")

'''
Part 2: Convert LiDAR points from body to camera 2 frame
'''

# Answer 2.1
C_BC = rotation_matrix_y(90)
print_func(C_BC, "C_BC")
T_BC = transformation_matrix(C_BC, 2.82, 0.11, 1.06)
print_func(T_BC, "T_BC")

# Answer 2.2
pose_body_test_2 = np.array([[4.47], [-0.206], [0.731]])
T_CB = np.linalg.inv(T_BC)
print_func(T_CB, "T_CB")
pose_camera_test = transform_point(T_CB, pose_body_test_2)
print_func(pose_camera_test, "pose_camera_test")

pose_normalized_test = normalized_image_plane_projection(pose_camera_test)
print_func(pose_normalized_test, "pose_normalized_test")

pose_distortion_test = lens_distortion(pose_normalized_test, -0.369, 0.197, 0.00135, 0.000568, -0.068)
print_func(pose_distortion_test, "pose_distortion_test")

pixel_test = pixel_coordinates(pose_distortion_test, 959.79, 956.93, 696.02, 224.18)
print_func(pixel_test, "pixel_test")

'''
Part 3: Project the points from the camera 2 frame to the image plane. You
may assume no lens distortion in the image. Remember to filter out points
where the projection does not lie within the image field, which is 1242x375.
'''

# Convert points from lidar frame to camera frame
pose_camera = transform_point(T_cam2_lidar, pose_lidar)
print_func(pose_camera, "pose_camera")

# Normalize points
pose_normalized = normalized_image_plane_projection(pose_camera)
print_func(pose_normalized, "pose_normalized")

# Factor Plum Distortion model
# pose_distortion = lens_distortion(pose_normalized, -0.369, 0.197, 0.00135, 0.000568, -0.068)
# print_func(pose_distortion, "pose_distortion")

# Convert points from camera frame to image frame
pixel = pixel_coordinates(pose_normalized, intrinsics[0,0], intrinsics[1,1], intrinsics[0,2], intrinsics[1,2])
print_func(pixel, "pixel")

# Create matrix where a row containing respective depth measurements wrt the body frame is added
pixel_with_depth = np.ones((np.shape(pixel)[0]+1, np.shape(pixel)[1]))
pixel_with_depth[0:2, :] = pixel
pixel_with_depth[2, :] = depth_lidar
print_func(pixel_with_depth, "pixel_with_depth")

# Delete points that are outside the camera's field of view
pixel_with_depth_reduced = pixel_coordinates_based_on_resolution(pixel_with_depth, 1242, 375)
print_func(pixel_with_depth_reduced, "pixel_with_depth_reduced")

# Part 4: Overlay the points on the image with the appropriate depth values.
# Use a colormap to show the difference between points' depths and remember to
# include a colorbar.
plt.figure()
plt.imshow(image)
plt.scatter(pixel_with_depth_reduced[0,:], pixel_with_depth_reduced[1,:], c=pixel_with_depth_reduced[2,:], cmap='viridis', s=1)
plt.colorbar()
plt.show()