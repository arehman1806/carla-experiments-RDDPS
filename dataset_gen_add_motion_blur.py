import os
import argparse
from PIL import Image, ImageFilter
from concurrent.futures import ThreadPoolExecutor

def add_motion_blur(image, level):
    if level == 1:
        return image.filter(ImageFilter.GaussianBlur(2))
    elif level == 2:
        return image.filter(ImageFilter.GaussianBlur(4))
    elif level == 3:
        return image.filter(ImageFilter.GaussianBlur(6))
    elif level == 4:
        return image.filter(ImageFilter.GaussianBlur(8))
    elif level == 5:
        return image.filter(ImageFilter.GaussianBlur(10))
    else:
        return image

def process_single_image(filename, record):
    rgb_folder = './recording/{}/rgb/'.format(record)
    instance_folder = './recording/{}/instance/'.format(record)
    filepath_rgb = os.path.join(rgb_folder, filename)
    filepath_instance = os.path.join(instance_folder, filename)

    # If the corresponding instance file doesn't exist, remove RGB and return
    if not os.path.exists(filepath_instance):
        os.remove(filepath_rgb)
        return
        
    # Process RGB files
    image_rgb = Image.open(filepath_rgb)
    frame = os.path.splitext(filename)[0]
    for index in range(1, 6):
        blurred_image = add_motion_blur(image_rgb, index)
        blurred_image.save(os.path.join(rgb_folder, "{}_{}.png".format(frame, index)))
    os.remove(filepath_rgb)

    # Process Instance files
    for index in range(1, 6):
        new_filename = "{}_{}.png".format(frame, index)
        os.system(f"cp {filepath_instance} {os.path.join(instance_folder, new_filename)}")
    os.remove(filepath_instance)

def process_images_multithreaded(record):
    rgb_folder = './recording/{}/rgb/'.format(record)
    
    # Using ThreadPoolExecutor to process images concurrently
    with ThreadPoolExecutor(max_workers=5) as executor:  # adjust max_workers as per your hardware
        executor.map(process_single_image, os.listdir(rgb_folder), [record] * len(os.listdir(rgb_folder)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images in RGB and Instance folders.")
    parser.add_argument('--record', type=str, help="Name of the record.", default="1692564886.98098")
    args = parser.parse_args()
    process_images_multithreaded(args.record)
