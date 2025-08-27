# LiDAR and Camera Fusion with YOLOv8 Segmentation

## Project Overview
This project demonstrates a **fusion of LiDAR 3D point clouds and camera images** to detect and visualize objects in 3D space. Specifically, the system detects cars in 2D images using YOLOv8 segmentation and maps these detections onto the LiDAR point cloud for 3D visualization.

## Objectives
- Detect cars in camera images using deep learning (YOLOv8).
- Project 2D detections into 3D LiDAR space.
- Visualize the LiDAR point cloud with color-coded detections and bounding boxes.
- Count and analyze points within each detected 3D bounding box.

## Methodology

### 1. Data Loading
- Synchronized camera images and LiDAR point clouds are loaded from the KITTI-360 dataset.
- Calibration files (`calib_cam_to_velo.txt` and `perspective.txt`) are used to transform coordinates between camera and LiDAR frames.

### 2. YOLO Segmentation
- YOLOv8 is applied to each image to segment objects (specifically cars).
- Segmentation masks are extracted and resized to match image dimensions.

### 3. 3D Projection
- LiDAR points are transformed to camera coordinates.
- Points falling inside YOLO-detected car masks are color-coded using a predefined color palette.

### 4. 3D Bounding Box Matching
- 3D bounding boxes from the dataset are transformed to LiDAR coordinates.
- Each bounding box is matched to the corresponding YOLO mask based on the overlap of points.
- Color-coded points inside each bounding box are counted, and statistics are printed for analysis.

### 5. Visualization
- Open3D is used to display the 3D point cloud with color-coded detections.
- Detected cars are highlighted with distinct colors, and 3D bounding boxes are drawn around them.

## Results
- Successfully detects and visualizes cars in 3D LiDAR space.
- Color-coded points indicate detected objects.
- Displays per-object point counts within 3D bounding boxes for quantitative analysis.

## Requirements
- Python 3.9+
- Libraries:
  - OpenCV (`cv2`)
  - NumPy (`numpy`)
  - Open3D (`open3d`)
  - Ultralytics YOLO (`ultralytics`)

## Usage
1. Update the dataset paths in the script:
   ```python
   base_dir = "PATH_TO_KITTI_PROJECT"
   image_dir = ...
   lidar_dir = ...
   bbox_dir = ...
   calib_cam_to_velo_path = ...
   intrinsics_path = ...

## OUTPUT
<img width="1408" height="376" alt="0000000100" src="https://github.com/user-attachments/assets/ff44152d-0f9e-4041-89e5-ea0b711f0802" />

![image first](https://github.com/user-attachments/assets/e75202c6-46c0-4a7b-919a-bace134ee65b)


<img width="1408" height="376" alt="0000001134" src="https://github.com/user-attachments/assets/ff6088e9-a158-4ede-b802-042f4adedad4" />

![image eight](https://github.com/user-attachments/assets/ea212e66-99d5-41c6-81ed-ed400d2223c4)

