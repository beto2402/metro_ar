import cv2


def get_resized_jpg(og_path):
    img = cv2.imread(og_path, cv2.IMREAD_UNCHANGED)
        
    dim = (128, 128)

    if len(img.shape) > 3:
        trans_mask = img[:,:,3] == 0

        # Replace areas of transparency with white and not transparent
        img[trans_mask] = [255, 255, 255, 255]

        # New image without alpha channel
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    
    return cv2.resize(img, dim, interpolation = cv2.INTER_AREA)


def resize_and_save(og_path, new_path):
    cv2.imwrite(new_path, get_resized_jpg(og_path))
