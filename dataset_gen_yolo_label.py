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
RAW_DATA_PATH = "{}/{}".format(ROOT_PATH, 'recording')
DATASET_PATH = "{}/{}".format(ROOT_PATH, 'dataset')

def gather_yolo_data(record_name: str, rgb_camera_name: str = 'rgb', semantic_camera_name: str = 'instance'):
    yolo_rawdata_df = pd.DataFrame()
    record_rawdata_path = f"{RAW_DATA_PATH}/{record_name}"
    rgb_image_path_list = sorted(glob.glob(f"{record_rawdata_path}/{rgb_camera_name}/*.png"))
    semantic_image_path_list = sorted(glob.glob(f"{record_rawdata_path}/{semantic_camera_name}/*.png"))

    # Filter only those frames for which both RGB and Semantic images exist
    rgb_basenames = {os.path.basename(p) for p in rgb_image_path_list}
    semantic_basenames = {os.path.basename(p) for p in semantic_image_path_list}

    common_basenames = rgb_basenames.intersection(semantic_basenames)
    rgb_image_path_list = [p for p in rgb_image_path_list if os.path.basename(p) in common_basenames]
    semantic_image_path_list = [p for p in semantic_image_path_list if os.path.basename(p) in common_basenames]

    yolo_rawdata_df['rgb_image_path'] = rgb_image_path_list
    yolo_rawdata_df['semantic_image_path'] = semantic_image_path_list
    yolo_rawdata_df['record_name'] = record_name

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
                    filename = frame_name
                    f.write(f'./images/{split_name}/{filename}.png\n')

        start = time.time()
        pool = ProcessPool()
        pool.starmap(self.process_frame_instance, rawdata_df.iterrows())
        pool.close()
        pool.join()
        print("cost: {:0<3f}s".format(time.time() - start))

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
            write_image(output_dir, frame_id, image_rgb, frame["split"])
            write_label(output_dir, frame_id, labels_all, frame["split"])

            
        write_yaml(output_dir)
        return

    

def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '--record', '-r',
        default="1692622994.599725",
        help='Rawdata Record ID. e.g. record_2022_0113_1337'
    )
    argparser.add_argument(
        '--vehicle', '-v',
        default='all',
        help='Vehicle name. e.g. `vehicle.tesla.model3_1`. Default to all vehicles. '
    )
    argparser.add_argument(
        '--rgb_camera', '-c',
        default='rgb',
        help='Camera name. e.g. image_2'
    )
    argparser.add_argument(
        '--semantic_camera', '-s',
        default='instance',
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
    rawdata_df = gather_yolo_data(args.record,
                                    args.rgb_camera,
                                    args.semantic_camera)
    # print("Process {} - {}".format(record_name, vehicle_name))
    yolo_label_tool.process(rawdata_df)


if __name__ == '__main__':
    main()

