import os
import cv2
import numpy as np
from carla_detect import CarlaObjectDetector
import pandas as pd

def bbox_iou(boxA, boxB):
    # Determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection rectangle
    inter_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # Compute the area of both bounding boxes
    boxA_area = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxB_area = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # Compute the IoU
    iou = inter_area / float(boxA_area + boxB_area - inter_area)

    return iou
def get_file_path(base_dir, filename):
    """
    Returns the path of the file, whether it's in 'train' or 'val'.
    """
    for subdir in ['train', 'val']:
        if os.path.exists(os.path.join(base_dir, subdir, filename)):
            return os.path.join(base_dir, subdir, filename)
    return None
# Read the CSV file using pandas
df = pd.read_csv('./carla/states.csv')

# Base directories
image_base_dir = './carla/images/'
label_base_dir = './carla/labels/'

# Initialize CarlaObjectDetector
detector = CarlaObjectDetector()

# Define the IoU threshold
iou_threshold = 0.5  # Adjust as needed

# New column for detected
df['detected'] = 0  # Initializing with 0

# List all the image files in the directory
# image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]

for index, row in df.iterrows():
    image_file = row['image_file_name'] + '.png'
    image_path = get_file_path(image_base_dir, image_file)
    if image_path:
        image = cv2.imread(image_path)
        
        # Extract ground truth boxes from the label
        label_file = image_file.replace('.png', '.txt')
        label_path = get_file_path(label_base_dir, label_file)

        ground_truth_boxes = []
        if label_path:
            with open(label_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    class_id, x_center, y_center, width, height = map(float, parts)
                    
                    x1 = int((x_center - width / 2) * image.shape[1])
                    y1 = int((y_center - height / 2) * image.shape[0])
                    x2 = int((x_center + width / 2) * image.shape[1])
                    y2 = int((y_center + height / 2) * image.shape[0])

                    ground_truth_boxes.append([x1, y1, x2, y2])

            # Get predicted boxes from the detector
            predicted_boxes, _ = detector.detect(image)

            # Calculate IoU for each predicted box with each ground truth box
            max_iou_for_image = 0
            for pred_box in predicted_boxes:
                pred_coords = pred_box[:4]
                for gt_box in ground_truth_boxes:
                    iou = bbox_iou(pred_coords, gt_box)
                    max_iou_for_image = max(max_iou_for_image, iou)

            # Update the 'detected' column based on the IoU
            df.at[index, 'detected'] = 1 if max_iou_for_image > iou_threshold else 0

# Save the updated dataframe to a new CSV file
df.to_csv('surrogate_risk_model_dataset.csv', index=False)