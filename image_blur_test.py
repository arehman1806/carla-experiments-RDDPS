import cv2
import numpy as np
import argparse

def alternative_radial_blur(img, blur=0.02, iterations=5):
    w, h = img.shape[:2]

    center_x = w / 2
    center_y = h / 2
    # blur = 0.02
    iterations = 5

    growMapx = np.tile(np.arange(h) + ((np.arange(h) - center_x)*blur), (w, 1)).astype(np.float32)
    shrinkMapx = np.tile(np.arange(h) - ((np.arange(h) - center_x)*blur), (w, 1)).astype(np.float32)
    growMapy = np.tile(np.arange(w) + ((np.arange(w) - center_y)*blur), (h, 1)).transpose().astype(np.float32)
    shrinkMapy = np.tile(np.arange(w) - ((np.arange(w) - center_y)*blur), (h, 1)).transpose().astype(np.float32)

    for i in range(iterations):
        tmp1 = cv2.remap(img, growMapx, growMapy, cv2.INTER_LINEAR)
        tmp2 = cv2.remap(img, shrinkMapx, shrinkMapy, cv2.INTER_LINEAR)
        img = cv2.addWeighted(tmp1, 0.5, tmp2, 0.5, 0)
    return img

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply alternative radial blur to an image.")
    parser.add_argument('--image', type=str, help="Path to the image you want to process.", required=False, default="2161.png")
    parser.add_argument('--blur', type=float, help="Blur radius per pixel from center. Default=0.02.", default=0.03)
    parser.add_argument('--iterations', type=int, help="Number of blur iterations. Default=5.", default=5)
    parser.add_argument('--output', type=str, help="Path to save the blurred image.", required=False, default="./2161_blurred_0.03.png")
    args = parser.parse_args()
    
    image = cv2.imread(args.image)
    blurred_image = alternative_radial_blur(image, args.blur, args.iterations)
    cv2.imwrite(args.output, blurred_image)
