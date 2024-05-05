from tt_constants import RENAMEDS_PATH, RESIZEDS_PATH, RESIZED_GS_PATH
from statistics import mean, stdev
import os
import cv2
from tt_utils import create_output_directory


def get_stats(base_path, renamed_rads_names):
    heights = []
    widths = []    
    
    # Read each image, get height and width and add them to heights/widhts lists
    for renamed_rad_path in renamed_rads_names:
        current_image_path = f"{base_path}/{renamed_rad_path}"
        img = cv2.imread(current_image_path)

        height, width, _ = img.shape

        heights.append(height)
        widths.append(width)


    max_height = max(heights)
    min_height = min(heights)
    mean_height = mean(heights)
    stdev_height = stdev(heights)

    print(f"Max height: {max_height}")
    print(f"Min height: {min_height}")
    print(f"Mean height: {mean_height}")
    print(f"Stdev height: {stdev_height}")
    print("")

    max_width = max(widths)
    min_width = min(widths)
    mean_width = mean(widths)
    stdev_width = stdev(widths)
    print(f"Max width: {max_width}")
    print(f"Min width: {min_width}")
    print(f"Mean width: {mean_width}")
    print(f"Stdev width: {stdev_width}")
    print("")


    return {
        "height": {
            "max": max_height,
            "min": min_height,
            "mean": mean_height,
            "stdev": stdev_height,
        },
        "width": {
            "max": max_width,
            "min": min_width,
            "mean": mean_width,
            "stdev": stdev_width,
        }
    }




def resize_rads(grayscale=False):
    output_path = RESIZEDS_PATH

    if grayscale:
        output_path = RESIZED_GS_PATH
        
    create_output_directory(output_path)
        

    renamed_rads_names = sorted(os.listdir(RENAMEDS_PATH))

    stats = get_stats(RENAMEDS_PATH, renamed_rads_names)

    new_width = 1350
    new_height = 772
    # width x height
    new_dim = (new_width, new_height)


    for rad_name in renamed_rads_names:
        current_image_path = f"{RENAMEDS_PATH}/{rad_name}"

        resized_image_path = f"{output_path}/{rad_name}"

        resize_and_save(current_image_path, resized_image_path, new_dim, grayscale=grayscale)


    resized_rads_names = sorted(os.listdir(output_path))

    get_stats(output_path, resized_rads_names)


def get_resized_jpg(og_path, dim):
    img = cv2.imread(og_path, cv2.IMREAD_UNCHANGED)
        

    if len(img.shape) > 3:
        trans_mask = img[:,:,3] == 0

        # Replace areas of transparency with white and not transparent
        img[trans_mask] = [255, 255, 255, 255]

        # New image without alpha channel
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    
    return cv2.resize(img, dim, interpolation = cv2.INTER_AREA)


def resize_and_save(og_path, new_path, new_dimension, grayscale=False):
    resized_img = get_resized_jpg(og_path, new_dimension)
    
    if grayscale:
        resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

    return cv2.imwrite(new_path, resized_img)


resize_rads(grayscale=True)

