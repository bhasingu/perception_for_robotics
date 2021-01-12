import os
import sys

import cv2 as cv
import numpy as np
import kitti_dataHandler
from matplotlib import pyplot as plt

def main():

    ################
    # Options
    ################
    # Input dir and output dir for train
    disp_dir_train = 'data/train/disparity'
    output_dir_train = 'data/train/est_depth'
    calib_dir_train = 'data/train/calib'
    sample_list_train = ['000001', '000002', '000003', '000004', '000005', '000006', '000007', '000008', '000009', '000010']
    # Input dir and output dir for test
    disp_dir_test = 'data/test/disparity'
    output_dir_test = 'data/test/est_depth'
    calib_dir_test = 'data/test/calib'
    sample_list_test = ['000011', '000012', '000013', '000014', '000015']
    ################

    for sample_name in (sample_list_train):
        # Read disparity map
        disp_map_path = disp_dir_train +'/' + sample_name + '.png'
        disp_map = cv.imread(disp_map_path, 0)
        # plt.imshow(disp_map, 'gray')
        # plt.show()
        # Read calibration info
        frame_calib = kitti_dataHandler.read_frame_calib(calib_dir_train + '/' +  sample_name + '.txt')
        stereo_calib = kitti_dataHandler.get_stereo_calibration(frame_calib.p2, frame_calib.p3)
        # Calculate depth (z = f*B/disp)
        num_u = np.shape(disp_map)[1]
        num_v = np.shape(disp_map)[0]
        depth_map = np.zeros((num_v, num_u))
        for i in range(num_u):
            for j in range(num_v):
                if disp_map[j, i] == 0:
                    continue
                else:
                    depth_map[j, i] = (1.0 / disp_map[j, i]) * stereo_calib.baseline * stereo_calib.f
        # Discard pixels past 80m
        for i in range(num_u):
            for j in range(num_v):
                if depth_map[j, i] > 80 or depth_map[j, i] < 10:
                    depth_map[j, i] = 0.0
        # Save depth map
        cv.imwrite(output_dir_train + '/depth_map_' + sample_name + '.jpg',depth_map)
    
    for sample_name in (sample_list_test):
        # Read disparity map
        disp_map_path = disp_dir_test +'/' + sample_name + '.png'
        disp_map = cv.imread(disp_map_path, 0)
        # plt.imshow(disp_map, 'gray')
        # plt.show()
        # Read calibration info
        frame_calib = kitti_dataHandler.read_frame_calib(calib_dir_test + '/' +  sample_name + '.txt')
        stereo_calib = kitti_dataHandler.get_stereo_calibration(frame_calib.p2, frame_calib.p3)
        # Calculate depth (z = f*B/disp)
        num_u = np.shape(disp_map)[1]
        num_v = np.shape(disp_map)[0]
        depth_map = np.zeros((num_v, num_u))
        for i in range(num_u):
            for j in range(num_v):
                if disp_map[j, i] == 0:
                    continue
                else:
                    depth_map[j, i] = (1.0 / disp_map[j, i]) * stereo_calib.baseline * stereo_calib.f
        # Discard pixels past 80m
        for i in range(num_u):
            for j in range(num_v):
                if depth_map[j, i] > 80 or depth_map[j, i] < 10:
                    depth_map[j, i] = 0.0
        # Save depth map
        cv.imwrite(output_dir_test + '/depth_map_' + sample_name + '.jpg',depth_map)


if __name__ == '__main__':
    main()
