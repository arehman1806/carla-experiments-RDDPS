import os
import argparse
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading

def alternative_radial_blur(img, blur=0.02, iterations=5):
    w, h = img.shape[:2]
    center_x = w / 2
    center_y = h / 2

    growMapx = np.tile(np.arange(h) + ((np.arange(h) - center_x)*blur), (w, 1)).astype(np.float32)
    shrinkMapx = np.tile(np.arange(h) - ((np.arange(h) - center_x)*blur), (w, 1)).astype(np.float32)
    growMapy = np.tile(np.arange(w) + ((np.arange(w) - center_y)*blur), (h, 1)).transpose().astype(np.float32)
    shrinkMapy = np.tile(np.arange(w) - ((np.arange(w) - center_y)*blur), (h, 1)).transpose().astype(np.float32)

    for i in range(iterations):
        tmp1 = cv2.remap(img, growMapx, growMapy, cv2.INTER_LINEAR)
        tmp2 = cv2.remap(img, shrinkMapx, shrinkMapy, cv2.INTER_LINEAR)
        img = cv2.addWeighted(tmp1, 0.5, tmp2, 0.5, 0)
    return img

def process_single_image(filename, record, counter, total, lock):
    rgb_folder = './recording/{}/rgb/'.format(record)
    instance_folder = './recording/{}/instance/'.format(record)
    filepath_rgb = os.path.join(rgb_folder, filename)
    filepath_instance = os.path.join(instance_folder, filename)

    # If the corresponding instance file doesn't exist, remove RGB and return
    if not os.path.exists(filepath_instance):
        os.remove(filepath_rgb)
        return
        
    # Process RGB files using cv2
    image_rgb = cv2.imread(filepath_rgb)
    frame = os.path.splitext(filename)[0]
    for index in range(0, 3):
        # Use the alternative_radial_blur method
        blurred_image = alternative_radial_blur(image_rgb, blur=(0.02/3)*index)
        cv2.imwrite(os.path.join(rgb_folder, "{}_{}.png".format(frame, index)), blurred_image)
    os.remove(filepath_rgb)

    # Process Instance files
    for index in range(0, 3):
        new_filename = "{}_{}.png".format(frame, index)
        os.system(f"cp {filepath_instance} {os.path.join(instance_folder, new_filename)}")
    os.remove(filepath_instance)
    with lock:
        counter[0] += 1
        if counter[0] % 100 == 0:
            print(f"Processed {counter[0]}/{total} files")

def process_images_multithreaded(record):
    rgb_folder = './recording/{}/rgb/'.format(record)
    instance_folder = './recording/{}/instance/'.format(record)
    
    rgb_files = set(os.listdir(rgb_folder))
    instance_files = set(os.listdir(instance_folder))
    
    # Find out images that are in 'instance' but not in 'rgb' and remove them
    extra_instance_files = instance_files - rgb_files
    for file in extra_instance_files:
        os.remove(os.path.join(instance_folder, file))
    
    counter = [0]  # using list as a mutable container for the integer
    total_files = len(os.listdir(rgb_folder))
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=5) as executor:  # adjust max_workers as per your hardware
        executor.map(process_single_image, os.listdir(rgb_folder), [record] * total_files, [counter] * total_files, [total_files] * total_files, [lock] * total_files)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images in RGB and Instance folders.")
    parser.add_argument('--record', type=str, help="Name of the record.", default="1692622994.599725")
    args = parser.parse_args()
    process_images_multithreaded(args.record)
