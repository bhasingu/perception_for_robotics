import os
import sys

import cv2 as cv
import numpy as np
import kitti_dataHandler
from part2_yolo import *
import pdb


def main():

    ################
    # Options
    ################
    # Input dir and output dir
    depth_dir_train = 'data/train/gt_depth'
    gt_seg_dir_train = 'data/train/gt_segmentation'
    label_dir_train = 'data/train/gt_labels'
    output_dir_train = 'data/train/est_segmentation'
    sample_list_train = ['000001', '000002', '000003', '000004', '000005', '000006', '000007', '000008', '000009', '000010']
    # sample_list_train = ['000001']
    

    depth_dir_test = 'data/test/gt_depth'
    label_dir_test = 'data/test/gt_labels'
    output_dir_test = 'data/test/est_segmentation'
    sample_list_test = ['000011', '000012', '000013', '000014', '000015']
    ################
    precision_list_train = []
    recall_list_train = []
    precision_list_test = []
    recall_list_test = []
    for sample_name in sample_list_train:
    	# Read depth map
        est_depth_path = "data/train/est_depth/depth_map_" + sample_name + ".jpg"
        est_depth = cv.imread(est_depth_path, 0)
        seg_mask = np.zeros((np.shape(est_depth)[0], np.shape(est_depth)[1]), dtype='int')
        for u in range(np.shape(est_depth)[1]):
            for v in range(np.shape(est_depth)[0]):
                seg_mask[v, u] = 255
        dist_threshold = 8.2
        # cv.imshow("est_depth", est_depth)
        # cv.waitKey(0)
        # Discard depths less than 10cm from the camera

        # Read 2d bbox
        image_path_train = 'data/train/left/' + sample_name + '.png'
        output_dir_box_train = 'data/train/bounding_box'
        idxs, classIDs, boxes = yolo(sample_name, image_path_train, output_dir_box_train)
        # For each bbox
        for idx in idxs:
            sum_depth = 0
            num_pixels = 0
            if classIDs[idx[0]] == 2: # class id of 2 is a car
                print(boxes[idx[0]])
                if boxes[idx[0]][0] < 0:
                    u_start_index = 0
                else:
                    u_start_index = boxes[idx[0]][0]
                if boxes[idx[0]][0] + boxes[idx[0]][2] > np.shape(est_depth)[1]:
                    u_end_index = np.shape(est_depth)[1]
                else:
                    u_end_index = boxes[idx[0]][0] + boxes[idx[0]][2]
                if boxes[idx[0]][1] < 0:
                    v_start_index = 0
                else:
                    v_start_index = boxes[idx[0]][1]
                if boxes[idx[0]][1] + boxes[idx[0]][3] > np.shape(est_depth)[0]:
                    v_end_index = np.shape(est_depth)[0]
                else:
                    v_end_index = boxes[idx[0]][1] + boxes[idx[0]][3]

                for u in range(u_start_index, u_end_index):
                    for v in range(v_start_index, v_end_index):
                        # Estimate the average depth of the objects
                        sum_depth += est_depth[v, u]
                        num_pixels += 1
                avg_depth_box = (sum_depth / num_pixels)
                # print("avg_depth")
                # print(avg_depth_box)
                # pdb.set_trace()
            for u in range(u_start_index, u_end_index):
                for v in range(v_start_index, v_end_index):
                    # Estimate the average depth of the objects
                    # print("depth value")
                    # print(est_depth[v, u])
                    if (est_depth[v, u] < avg_depth_box + dist_threshold) and (est_depth[v, u] > avg_depth_box - dist_threshold):
                        seg_mask[v, u] = 0
        cv2.imwrite(output_dir_train + '/' + sample_name + '.jpg',seg_mask)
        gt_seg_path = gt_seg_dir_train + '/' + sample_name + ".png"
        gt_seg = cv.imread(gt_seg_path, 0)
        # print("values in gt_seg")
        # print(np.unique(gt_seg))
        # print("values in seg_mask")
        # print(np.unique(seg_mask))
        TP = 0
        FP = 0
        FN = 0
        for u in range(np.shape(gt_seg)[1]):
            for v in range(np.shape(gt_seg)[0]):
                if gt_seg[v, u] < 255 and seg_mask[v, u] == 0:
                    TP += 1
                if gt_seg[v, u] == 255 and seg_mask[v, u] == 0:
                    FP += 1
                if gt_seg[v, u] < 255 and seg_mask[v, u] == 255:
                    FN += 1
        precision_list_train.append(TP / (TP + FP))
        recall_list_train.append(TP / (TP + FN))
    print("Precision list")
    print(precision_list_train)
    print("Recall list")
    print(recall_list_train)
    print("Avg Precision")
    print(sum(precision_list_train)/len(precision_list_train))
    print("Avg Recall")
    print(sum(recall_list_train)/len(recall_list_train))
            # Find the pixels within a certain distance from the centroid

        # Save the segmentation mask
    for sample_name in sample_list_test:
    	# Read depth map
        est_depth_path = "data/test/est_depth/depth_map_" + sample_name + ".jpg"
        est_depth = cv.imread(est_depth_path, 0)
        seg_mask = np.zeros((np.shape(est_depth)[0], np.shape(est_depth)[1]), dtype='int')
        for u in range(np.shape(est_depth)[1]):
            for v in range(np.shape(est_depth)[0]):
                seg_mask[v, u] = 255
        dist_threshold = 8.2
        # cv.imshow("est_depth", est_depth)
        # cv.waitKey(0)
        # Discard depths less than 10cm from the camera

        # Read 2d bbox
        image_path_test = 'data/test/left/' + sample_name + '.png'
        output_dir_box_test = 'data/test/bounding_box'
        idxs, classIDs, boxes = yolo(sample_name, image_path_test, output_dir_box_test)
        # For each bbox
        for idx in idxs:
            sum_depth = 0
            num_pixels = 0
            if classIDs[idx[0]] == 2: # class id of 2 is a car
                print(boxes[idx[0]])
                if boxes[idx[0]][0] < 0:
                    u_start_index = 0
                else:
                    u_start_index = boxes[idx[0]][0]
                if boxes[idx[0]][0] + boxes[idx[0]][2] > np.shape(est_depth)[1]:
                    u_end_index = np.shape(est_depth)[1]
                else:
                    u_end_index = boxes[idx[0]][0] + boxes[idx[0]][2]
                if boxes[idx[0]][1] < 0:
                    v_start_index = 0
                else:
                    v_start_index = boxes[idx[0]][1]
                if boxes[idx[0]][1] + boxes[idx[0]][3] > np.shape(est_depth)[0]:
                    v_end_index = np.shape(est_depth)[0]
                else:
                    v_end_index = boxes[idx[0]][1] + boxes[idx[0]][3]

                for u in range(u_start_index, u_end_index):
                    for v in range(v_start_index, v_end_index):
                        # Estimate the average depth of the objects
                        sum_depth += est_depth[v, u]
                        num_pixels += 1
                avg_depth_box = (sum_depth / num_pixels)
                # print("avg_depth")
                # print(avg_depth_box)
                # pdb.set_trace()
            for u in range(u_start_index, u_end_index):
                for v in range(v_start_index, v_end_index):
                    # Estimate the average depth of the objects
                    if (est_depth[v, u] < avg_depth_box + dist_threshold) and (est_depth[v, u] > avg_depth_box - dist_threshold):
                        seg_mask[v, u] = 0
        cv2.imwrite(output_dir_test + '/' + sample_name + '.jpg',seg_mask)
            # Find the pixels within a certain distance from the centroid

        # Save the segmentation mask
if __name__ == '__main__':
    main()
