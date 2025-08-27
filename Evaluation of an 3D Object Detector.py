import os
import cv2
import json
import numpy as np
import open3d as o3d
from ultralytics import YOLO

# === PATHS ===
base_dir = r'E:\hrushi\PROJECTS\Lidar Radar\PROJECT\KITTI-360_sample'
image_dir = os.path.join(base_dir, r'data_2d_raw\2013_05_28_drive_0000_sync\image_00\data_rect')
lidar_dir = os.path.join(base_dir, r'data_3d_raw\2013_05_28_drive_0000_sync\velodyne_points\data')
bbox_dir = os.path.join(base_dir, r'bboxes_3D_cam0')
calib_cam_to_velo_path = os.path.join(base_dir, r'calibration\calib_cam_to_velo.txt')
intrinsics_path = os.path.join(base_dir, r'calibration\perspective.txt')

# === COLOR MAP ===
palette = [
    (255, 0, 0),      # red
    (0, 255, 0),      # green
    (0, 0, 255),      # blue
    (255, 255, 0),    # yellow
    (0, 255, 255),    # cyan
    (255, 0, 255)     # magenta
]

color_name_map = {
    (255, 0, 0): "red",
    (0, 255, 0): "green",
    (0, 0, 255): "blue",
    (255, 255, 0): "yellow",
    (0, 255, 255): "cyan",
    (255, 0, 255): "magenta"
}

# === CALIBRATION ===
def load_transformation(calib_file):
    with open(calib_file, 'r') as f:
        values = list(map(float, f.read().strip().split()))
        T = np.array(values, dtype=np.float32).reshape(3, 4)
        return np.vstack((T, [0, 0, 0, 1]))

def load_intrinsics(perspective_path):
    with open(perspective_path, 'r') as f:
        for line in f:
            if line.startswith('P_rect_00:'):
                vals = list(map(float, line.strip().split()[1:]))
                return np.array(vals, dtype=np.float32).reshape(3, 4)
    raise ValueError("P_rect_00 not found.")

# === 3D BBOX LINES ===
def create_bbox_lines(corners, color):
    lines = [[0,5],[1,4],[2,7],[3,6],[0,1],[1,3],[3,2],[2,0],[4,5],[5,7],[7,6],[6,4]]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color] * len(lines))
    return line_set

# === SETUP ===
T_cam_to_velo = load_transformation(calib_cam_to_velo_path)
T_velo_to_cam = np.linalg.inv(T_cam_to_velo)
P_rect = load_intrinsics(intrinsics_path)
model = YOLO("yolov8x-seg.pt")  # Updated model

# === BATCH VISUALIZATION ===
image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])

for image_file in image_files:
    try:
        print(f"\nVisualizing {image_file}...")

        img_id = os.path.splitext(image_file)[0]

        # === SUPPORT BOTH 3-DIGIT AND 4-DIGIT BBOX FILENAMES ===
        possible_ids = [img_id, img_id[-4:], img_id[-3:]]
        bbox_file = None

        for pid in possible_ids:
            candidate = f"BBoxes_{pid}.json"
            candidate_path = os.path.join(bbox_dir, candidate)
            if os.path.exists(candidate_path):
                bbox_file = candidate
                break

        if bbox_file is None:
            print(f"BBox file not found for image {img_id}")
            continue

        bbox_path = os.path.join(bbox_dir, bbox_file)
        image_path = os.path.join(image_dir, image_file)
        lidar_path = os.path.join(lidar_dir, f"{img_id}.bin")

        # Load data
        image = cv2.imread(image_path)
        h, w = image.shape[:2]
        lidar_points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)[:, :3]

        lidar_hom = np.hstack((lidar_points, np.ones((lidar_points.shape[0], 1))))
        points_cam = (T_velo_to_cam @ lidar_hom.T).T

        in_front = points_cam[:, 2] > 0
        points_cam = points_cam[in_front]
        lidar_points = lidar_points[in_front]

        proj = P_rect @ points_cam.T
        u = proj[0] / proj[2]
        v = proj[1] / proj[2]

        valid_mask = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        u = u[valid_mask].astype(int)
        v = v[valid_mask].astype(int)
        lidar_points = lidar_points[valid_mask]
        points_cam = points_cam[valid_mask]

        colors = np.full((len(lidar_points), 3), 128, dtype=np.uint8)

        # === YOLO SEGMENTATION ===
        results = model(image)
        masks = results[0].masks
        classes = results[0].boxes.cls.cpu().numpy() if results[0].boxes else []

        yolo_car_masks = []
        yolo_car_colors = []

        if masks is not None:
            car_id = 0
            for i, cls in enumerate(classes):
                if int(cls) == 2:  # class 2 is 'car'
                    mask = masks.data[i].cpu().numpy()
                    mask_resized = cv2.resize(mask, (w, h))
                    mask_binary = (mask_resized > 0.5).astype(np.uint8)

                    indices = []
                    for pt_idx, (ui, vi) in enumerate(zip(u, v)):
                        if mask_binary[vi, ui]:
                            indices.append(pt_idx)
                            colors[pt_idx] = palette[car_id % len(palette)]

                    yolo_car_masks.append(set(indices))
                    yolo_car_colors.append(palette[car_id % len(palette)])
                    car_id += 1

        # === MATCH 3D BOX TO MASK ===
        with open(bbox_path, 'r') as f:
            bbox_data = json.load(f)

        assigned_bboxes = []
        used_boxes = set()

        for mask_indices, color in zip(yolo_car_masks, yolo_car_colors):
            best_match = None
            best_overlap = 0

            for idx, box in enumerate(bbox_data):
                if idx in used_boxes:
                    continue

                corners_cam0 = np.array(box["corners_cam0"])
                corners_hom = np.hstack((corners_cam0, np.ones((8, 1))))
                corners_lidar = (T_cam_to_velo @ corners_hom.T).T[:, :3]

                min_corner = corners_lidar.min(axis=0)
                max_corner = corners_lidar.max(axis=0)

                inside_mask = np.all((lidar_points >= min_corner) & (lidar_points <= max_corner), axis=1)
                inside_indices = set(np.where(inside_mask)[0])

                overlap = len(inside_indices & mask_indices)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match = corners_lidar
                    best_idx = idx

            if best_match is not None:
                used_boxes.add(best_idx)
                bbox = create_bbox_lines(best_match, np.array(color) / 255.0)
                assigned_bboxes.append(bbox)

                # Count color-matched points inside bbox
                min_corner = best_match.min(axis=0)
                max_corner = best_match.max(axis=0)
                inside_mask = np.all((lidar_points >= min_corner) & (lidar_points <= max_corner), axis=1)
                inside_indices = np.where(inside_mask)[0]
                color_arr = np.array(color, dtype=np.uint8)
                color_matches = np.all(colors[inside_indices] == color_arr, axis=1)
                count = np.sum(color_matches)

                color_tuple = tuple(color)
                color_name = color_name_map.get(color_tuple, "unknown")
                print(f"  Box {best_idx:02d}: {count} points ({color_name})")

        # === SHOW VISUALIZATION ===
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(lidar_points)
        pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)

        o3d.visualization.draw_geometries([pcd] + assigned_bboxes)

    except Exception as e:
        print(f"Skipping {image_file} due to error: {e}")
