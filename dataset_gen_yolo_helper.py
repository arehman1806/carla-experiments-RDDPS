#!/usr/bin/python3
import argparse
import os
import time

import cv2
import glob
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml


class YoloConfig:
    rectangle_pixels_min = 50
    color_pixels_min = 30


LABEL_DATAFRAME = pd.DataFrame(columns=['raw_value', 'color', 'coco_names_index'],
                               data=[
                                    #  [ 4, (220, 20, 60), 0],
                                     [10, (0, 0, 142), 0],
                                    #  [18, (250, 170, 30), 9],
                                    #  [12, (220, 220,  0), 80],
                               ])

TL_LIGHT_LABEL = {'DEFAULT': 9,
                  'RED': 82,
                  'GREEN': 81}

# LABEL_COLORS = np.array([
#     # (220, 20, 60),   # Pedestrian
#     # (0, 0, 142),     # Vehicle
#     (220, 220, 0),   # TrafficSign -> COCO INDEX
#     (250, 170, 30),  # TrafficLight
# ])

COCO_NAMES = ['vehicle']


def decrease_brightness(img, value=30):
    h, s, v = cv2.split(img)
    lim = 0 + value
    v[v < lim] = lim
    v[v >= lim] -= lim
    final_hsv = cv2.merge((h, s, v))
    return final_hsv


def check_color(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_img = decrease_brightness(hsv, 80)

    red_min = np.array([0, 5, 150])
    red_max = np.array([15, 255, 255])
    red_min_2 = np.array([175, 5, 150])
    red_max_2 = np.array([180, 255, 255])

    yellow_min = np.array([25, 5, 150])
    yellow_max = np.array([35, 180, 255])

    green_min = np.array([35, 5, 150])
    green_max = np.array([90, 255, 255])

    red_thresh = cv2.inRange(hsv_img, red_min, red_max) + cv2.inRange(hsv_img, red_min_2, red_max_2)
    yellow_thresh = cv2.inRange(hsv_img, yellow_min, yellow_max)
    green_thresh = cv2.inRange(hsv_img, green_min, green_max)

    red_blur = cv2.medianBlur(red_thresh, 5)
    yellow_blur = cv2.medianBlur(yellow_thresh, 5)
    green_blur = cv2.medianBlur(green_thresh, 5)

    red = cv2.countNonZero(red_blur)
    yellow = cv2.countNonZero(yellow_blur)
    green = cv2.countNonZero(green_blur)

    light_color = max(red, green, yellow)
    if light_color > YoloConfig.color_pixels_min:
        if light_color == red:
            return TL_LIGHT_LABEL["RED"]
        elif light_color == green:
            return TL_LIGHT_LABEL["GREEN"]
        else:
            return TL_LIGHT_LABEL["DEFAULT"]
    else:
        return TL_LIGHT_LABEL["DEFAULT"]


def write_yaml(output_path):
    os.makedirs(output_path, exist_ok=True)
    dict_file = {
        'path': 'yolo_dataset',
        'train': 'images/train',
        'val': 'images/train',
        'test': 'images/test',
        'nc': len(COCO_NAMES),
        'names': COCO_NAMES
    }
    with open(f"{output_path}/yolov5_carla.yaml", 'w') as file:
        yaml.dump(dict_file, file)


def write_image(output_path: str, frame_id: str, image_rgb: np.array, vehicle="", split="train"):
    # print(f"attempting to write file at {image_dir}/{filename}.png")
    image_dir = f"{output_path}/images/{split}"
    os.makedirs(image_dir, exist_ok=True)
    filename = vehicle + frame_id
    
    cv2.imwrite(f"{image_dir}/{filename}.png", image_rgb)
    # print(f"have written the file at {image_dir}/{filename}.png")


def write_label(output_path: str, frame_id: str, labels: list, vehicle="", split="train"):
    label_dir = f"{output_path}/labels/{split}"
    os.makedirs(label_dir, exist_ok=True)
    filename = vehicle + frame_id
    with open(f"{label_dir}/{filename}.txt", "w") as f:
        for label in labels:
            f.write(label)
            f.write('\n')


def get_filename_from_fullpath(fullpath: str) -> str:
    filename = os.path.splitext(os.path.basename(fullpath))[0]
    return filename


def check_id(rgb_img_path, seg_img_path):
    img_name = get_filename_from_fullpath(rgb_img_path)
    seg_name = get_filename_from_fullpath(seg_img_path)
    if img_name != seg_name:
        print("Img name error: {} {}".format(img_name, seg_name))
        return False
    else:
        return True
