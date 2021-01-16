# perception_for_robotics

There are 3 packages in this repository:
- "coordinate_transforms_and_feature_projections"
- "feature_point_detection_and_correspondences"
- "3D_object_detection_and_instance_segmentation"

## Package - "coordinate_transforms_and_feature_projections"

### LiDAR range measurments transformed to camera frame and projected on image
![](coordinate_transforms_and_feature_projections/plot/final.jpeg)

## Package - "feature_point_detection_and_correspondences"

### SIFT keypoint matches between left and right stereo images after RANSAC outlier rejection
![](feature_point_detection_and_correspondences/results/q3/sift_matches_test_RANSAC_000011.jpg)
![](feature_point_detection_and_correspondences/results/q3/sift_matches_test_RANSAC_000012.jpg)
![](feature_point_detection_and_correspondences/results/q3/sift_matches_test_RANSAC_000013.jpg)
![](feature_point_detection_and_correspondences/results/q3/sift_matches_test_RANSAC_000014.jpg)
![](feature_point_detection_and_correspondences/results/q3/sift_matches_test_RANSAC_000015.jpg)

## Package - "3D_object_detection_and_instance_segmentation"

### Estimated depth using provided disparity maps between left and right stereo images
![](3D_object_detection_and_instance_segmentation/data/test/est_depth/depth_map_000011.jpg)
![](3D_object_detection_and_instance_segmentation/data/test/est_depth/depth_map_000012.jpg)
![](3D_object_detection_and_instance_segmentation/data/test/est_depth/depth_map_000013.jpg)
![](3D_object_detection_and_instance_segmentation/data/test/est_depth/depth_map_000014.jpg)
![](3D_object_detection_and_instance_segmentation/data/test/est_depth/depth_map_000015.jpg)

### Bounding boxes around cars in left images using YOLOv3
![](3D_object_detection_and_instance_segmentation/data/test/est_segmentation/000011.jpg)
![](3D_object_detection_and_instance_segmentation/data/test/est_segmentation/000012.jpg)
![](3D_object_detection_and_instance_segmentation/data/test/est_segmentation/000013.jpg)
![](3D_object_detection_and_instance_segmentation/data/test/est_segmentation/000014.jpg)
![](3D_object_detection_and_instance_segmentation/data/test/est_segmentation/000015.jpg)

### Instance segmentation on left stereo images for every detected car
