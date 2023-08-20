#!/usr/bin/python3
import glob
import os
import sys
from pathlib import Path
from multiprocessing import Pool as ProcessPool

import pandas as pd

sys.path.append(Path(__file__).resolve().parent.parent.as_posix())  # repo path
from label_tools.yolov5.yolov5_helper import *


ROOT_PATH = Path(__file__).parent.as_posix()
RAW_DATA_PATH = "{}/{}".format(ROOT_PATH, 'raw_data')
DATASET_PATH = "{}/{}".format(ROOT_PATH, 'dataset')

def gather_yolo_data(record_name: str, vehicle_name: str, rgb_camera_name: str, semantic_camera_name: str):
    yolo_rawdata_df = pd.DataFrame()
    vehicle_rawdata_path = f"{RAW_DATA_PATH}/{record_name}/{vehicle_name}"
    rgb_image_path_list = sorted(glob.glob(f"{vehicle_rawdata_path}/{rgb_camera_name}/*.png"))
    semantic_image_path_list = sorted(glob.glob(f"{vehicle_rawdata_path}/{semantic_camera_name}/*.png"))
    yolo_rawdata_df['rgb_image_path'] = rgb_image_path_list
    yolo_rawdata_df['semantic_image_path'] = semantic_image_path_list
    yolo_rawdata_df['record_name'] = record_name
    yolo_rawdata_df['vehicle_name'] = vehicle_name

    # Adding train/valid/test splits
    np.random.seed(42)
    yolo_rawdata_df['random_number'] = np.random.rand(len(yolo_rawdata_df))
    yolo_rawdata_df['split'] = yolo_rawdata_df['random_number'].apply(lambda x: 'train' if x < 0.7 else 'val' if x < 0.9 else 'test')
    yolo_rawdata_df = yolo_rawdata_df.drop('random_number', axis=1)
    return yolo_rawdata_df


class YoloLabelTool:
    def __init__(self):
        self.rec_pixels_min = 150
        self.color_pixels_min = 30
        self.debug = False

    def process(self, rawdata_df: pd.DataFrame):
        for split_name in rawdata_df["split"].unique():
            output_dir = f"{DATASET_PATH}"
            split_df = rawdata_df[rawdata_df['split'] == split_name]
            frame_names = split_df['rgb_image_path'].apply(get_filename_from_fullpath)
            with open(os.path.join(output_dir, f'{split_name}_yolo.txt'), 'a') as f:
                for i, frame_name in enumerate(frame_names):
                    filename = split_df["vehicle_name"].iloc[i] + frame_name
                    f.write(f'./images/{split_name}/{filename}.png\n')

        start = time.time()
        pool = ProcessPool()
        pool.starmap(self.process_frame_instance, rawdata_df.iterrows())
        pool.close()
        pool.join()
        print("cost: {:0<3f}s".format(time.time() - start))



        # start = time.time()
        # for index, frame in rawdata_df.iterrows():
        #     self.process_frame(index, frame)
        # print("cost: {:0<3f}s".format(time.time() - start))

    def process_frame_instance(self, index, frame):
        rgb_img_path = frame['rgb_image_path']
        seg_img_path = frame['semantic_image_path']
        success = check_id(rgb_img_path, seg_img_path)
        if not success:
            return

        output_dir = f"{DATASET_PATH}"
        frame_id = get_filename_from_fullpath(rgb_img_path)

        image_rgb = None
        image_seg = None
        image_rgb = cv2.imread(rgb_img_path, cv2.IMREAD_COLOR)
        image_seg = cv2.imread(seg_img_path, cv2.IMREAD_UNCHANGED)
        if image_rgb is None or image_seg is None:
            return

        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_RGBA2RGB)
        image_seg = cv2.cvtColor(image_seg, cv2.COLOR_BGRA2RGB)
        
        R, G, B = cv2.split(image_seg)

        # Create a binary mask where red == 10
        mask = np.where(R == 10, 255, 0)

        # The mask is a float array, but images are usually 8-bit, so convert it
        mask = mask.astype(np.uint8)

        # Mask the G and B channels
        G = np.where(mask == 255, G, 0)
        B = np.where(mask == 255, B, 0)

        # Stack G and B to create a 2D image for unique pairs
        GB = np.dstack((G, B))

        # Find unique pairs in the GB image
        unique_pairs = np.unique(GB.reshape(-1, GB.shape[-1]), axis=0)

        height, width, _ = image_rgb.shape
        labels_all = []

        # For each unique pair, draw a bounding box on the image
        for pair in unique_pairs:
            # Check if pair is not [0,0] (which is not a vehicle)
            if np.any(pair > 0):
                # Find the pixels that have this G-B pair
                pixels = np.where(np.all(GB == pair, axis=-1))
                
                # Get the bounding box coordinates
                x_min, x_max = np.min(pixels[1]), np.max(pixels[1])
                y_min, y_max = np.min(pixels[0]), np.max(pixels[0])

                if (x_max - x_min) * (y_max - y_min) < YoloConfig.rectangle_pixels_min:
                    continue
                
                # Add the bounding box coordinates and the label to the list of labels
                label_info = "{} {} {} {} {}".format(0,
                                                    float(x_min + ((x_max - x_min) / 2.0)) / width,
                                                    float(y_min + ((y_max - y_min) / 2.0)) / height,
                                                    float(x_max - x_min) / width,
                                                    float(y_max - y_min) / height)
                labels_all.append(label_info)

                # Debug: draw the bounding box
                # cv2.rectangle(image_rgb, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)

        if len(labels_all) > 0:
            write_image(output_dir, frame_id, image_rgb, frame["vehicle_name"], frame["split"])
            write_label(output_dir, frame_id, labels_all, frame["vehicle_name"], frame["split"])
            
        write_yaml(output_dir)
        return

    def process_frame(self, index, frame):
        rgb_img_path = frame['rgb_image_path']
        seg_img_path = frame['semantic_image_path']
        success = check_id(rgb_img_path, seg_img_path)
        if not success:
            return

        # output_dir = f"{DATASET_PATH}/{frame['record_name']}/{frame['vehicle_name']}/yolo"
        output_dir = f"{DATASET_PATH}"
        frame_id = get_filename_from_fullpath(rgb_img_path)

        image_rgb = None
        image_seg = None
        image_rgb = cv2.imread(rgb_img_path, cv2.IMREAD_COLOR)
        image_seg = cv2.imread(seg_img_path, cv2.IMREAD_UNCHANGED)
        if image_rgb is None or image_seg is None:
            return
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_RGBA2RGB)
        image_seg = cv2.cvtColor(image_seg, cv2.COLOR_BGRA2RGB)
        img_name = os.path.basename(rgb_img_path)
        height, width, _ = image_rgb.shape

        labels_all = []
        for index, label_info in LABEL_DATAFRAME.iterrows():
            seg_color = label_info['color']
            coco_id = label_info['coco_names_index']

            mask = (image_seg == seg_color)
            tmp_mask = (mask.sum(axis=2, dtype=np.uint8) == 3)
            mono_img = np.array(tmp_mask * 255, dtype=np.uint8)

            preview_img = image_rgb
            # self.preview_img = self.image_seg
            # cv2.imshow("seg", self.preview_img)
            # cv2.imshow("mono", mono_img)
            # cv2.waitKey()
            # bordered_image = cv2.copyMakeBorder(mono_img, top=10, bottom=10, left=10, right=10, 
            #                         borderType=cv2.BORDER_CONSTANT, value=[0,0,0])

            # contours, _ = cv2.findContours(bordered_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

            contours, hierarchy = cv2.findContours(mono_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            labels = []
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if w * h < YoloConfig.rectangle_pixels_min:
                    continue
                # cv2.rectangle(self.preview_img, (x, y), (x + w, y + h), (0, 255, 0), 1)
                # cv2.imshow("rect", self.preview_img)
                # cv2.waitKey()
                max_y, max_x, _ = image_rgb.shape
                
                # UNCOMMENT TO IGNORE BBOX THAT TOUCH THE EDGES
                # if y + h >= max_y or x + w >= max_x:
                #     continue

                if coco_id == TL_LIGHT_LABEL["DEFAULT"]:
                    coco_id = check_color(image_rgb[y:y + h, x:x + w, :])

                # DEBUG START
                # Draw label info to image
                cv2.putText(preview_img, COCO_NAMES[coco_id], (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 1)
                cv2.rectangle(image_rgb, (x, y), (x + w, y + h), (0, 255, 0), 1)
                # DEBUG END

                label_info = "{} {} {} {} {}".format(coco_id,
                                                     float(x + (w / 2.0)) / width,
                                                     float(y + (h / 2.0)) / height,
                                                     float(w) / width,
                                                     float(h) / height)
                labels.append(label_info)
                # cv2.imshow("result", self.preview_img)
                # cv2.imshow("test", self.preview_img[y:y+h, x:x+w, :])
                # cv2.waitKey()

            # First, generate all bounding boxes with corresponding labels
            # all_boxes = []
            # for index, cnt in enumerate(contours):
            #     x, y, w, h = cv2.boundingRect(cnt)
            #     if w * h < YoloConfig.rectangle_pixels_min or hierarchy[0][index][3] != -1:
            #         continue
            #     max_y, max_x, _ = image_rgb.shape
            #     all_boxes.append((x, y, w, h, index))

            # # Filter out nested bounding boxes
            # filtered_boxes = []
            # for box1 in all_boxes:
            #     x1, y1, w1, h1, index1 = box1
            #     if not any((x1 > x2) and (y1 > y2) and (x1 + w1 < x2 + w2) and (y1 + h1 < y2 + h2) for x2, y2, w2, h2, index2 in all_boxes if index1 != index2):
            #         filtered_boxes.append(box1)

            # # Now filtered_boxes contains only the non-nested boxes.
            # # We can process these further:
            # for x, y, w, h, index in filtered_boxes:
            #     if coco_id == TL_LIGHT_LABEL["DEFAULT"]:
            #         coco_id = check_color(image_rgb[y:y + h, x:x + w, :])

            #     # DEBUG START
            #     # Draw label info to image
            #     cv2.putText(preview_img, COCO_NAMES[coco_id], (x, y - 10),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 1)
            #     cv2.rectangle(image_rgb, (x, y), (x + w, y + h), (0, 255, 0), 1)
            #     # DEBUG END

            #     label_info = "{} {} {} {} {}".format(coco_id,
            #                                         float(x + (w / 2.0)) / width,
            #                                         float(y + (h / 2.0)) / height,
            #                                         float(w) / width,
            #                                         float(h) / height)
            #     labels.append(label_info)


            if len(labels) > 0:
                labels_all += labels

        if len(labels_all) > 0:

            # print("Label output path: {}".format(self.label_out_path))
            write_image(output_dir, frame_id, image_rgb, frame["vehicle_name"], frame["split"])
            write_label(output_dir, frame_id, labels_all, frame["vehicle_name"], frame["split"])
        write_yaml(output_dir)
        return


def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '--record', '-r',
        required=True,
        help='Rawdata Record ID. e.g. record_2022_0113_1337'
    )
    argparser.add_argument(
        '--vehicle', '-v',
        default='all',
        help='Vehicle name. e.g. `vehicle.tesla.model3_1`. Default to all vehicles. '
    )
    argparser.add_argument(
        '--rgb_camera', '-c',
        default='image_2',
        help='Camera name. e.g. image_2'
    )
    argparser.add_argument(
        '--semantic_camera', '-s',
        default='image_2_instance',
        help='Camera name. e.g. image_2_semantic'
    )

    args = argparser.parse_args()

    record_name = args.record
    if args.vehicle == 'all':
        vehicle_name_list = [os.path.basename(x) for x in
                             glob.glob('{}/{}/vehicle.*'.format(RAW_DATA_PATH, record_name))]
    else:
        vehicle_name_list = [args.vehicle]

    yolo_label_tool = YoloLabelTool()
    for vehicle_name in vehicle_name_list:
        rawdata_df = gather_yolo_data(args.record,
                                      vehicle_name,
                                      args.rgb_camera,
                                      args.semantic_camera)
        print("Process {} - {}".format(record_name, vehicle_name))
        yolo_label_tool.process(rawdata_df)


if __name__ == '__main__':
    main()

