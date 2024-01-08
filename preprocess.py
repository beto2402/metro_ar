import cv2
from PIL import Image

default_t_1 = 48
default_t_2 = 15


def get_resized_jpg(og_path):
    img = cv2.imread(og_path, cv2.IMREAD_UNCHANGED)
        
    dim = (144, 144)

    if len(img.shape) > 3:
        trans_mask = img[:,:,3] == 0

        # Replace areas of transparency with white and not transparent
        img[trans_mask] = [255, 255, 255, 255]

        # New image without alpha channel
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    
    return cv2.resize(img, dim, interpolation = cv2.INTER_AREA)


def resize_and_save(og_path, new_path, grayscale=False):
    resized_img = get_resized_jpg(og_path)
    
    if grayscale:
        resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

    return cv2.imwrite(new_path, resized_img)


def save_png_as_jpg(og_img_path, new_path):
    img = Image.open(og_img_path)

        
    # Create a new background image in white color
    background = Image.new(img.mode[:-1], img.size, (255, 255, 255))
    # Paste the original image onto the background
    background.paste(img, img.split()[-1])
    img = background

    # Convert the image to RGB mode if it's not
    
    img = img.convert('RGB')

    img.save(new_path)