import cv2
import numpy as np
import random
from tkinter import Tk
from tkinter.filedialog import askopenfilename


BRILLIANT = "Brillante"
PARTIALLY_CLOUDED = "Parcialmente nublado"
CLOUDY = "Nublado"


indexes = {
    BRILLIANT: list(range(1, 301)),
    PARTIALLY_CLOUDED: list(range(301, 601)),
    CLOUDY: list(range(601, 901))
}


def stack_images(scale, img_array):
    rows = len(img_array)
    cols = len(img_array[0])
    rowsAvailable = isinstance(img_array[0], list)
    width =img_array[0][0].shape[1]
    height = img_array[0][0].shape[0]
    if rowsAvailable:
        for x in range (0, rows):
            for y in range(0, cols):
                if img_array[x][y].shape[:2] == img_array[0][0].shape[:2]:
                    img_array[x][y] = cv2.resize(img_array[x][y], (0,0), None, scale, scale)
                else:
                    img_array[x][y] = cv2.resize(img_array[x][y], (img_array[0][0].shape[1], img_array[0][0].shape[0]), None, scale, scale)
                if len(img_array[x][y].shape) == 2: img_array[x][y] = cv2.cvtColor(img_array[x][y], cv2.COLOR_GRAY2BGR)
        image_blank = np.zeros((height, width, 3), np.uint8)
        hor = [image_blank]*rows
        hor_con = [image_blank] *rows
        for x in range (0,rows):
            hor[x] = np.hstack(img_array[x])
        ver = np.vstack(hor)
    else:
        for x in range(0,rows):
            if img_array[x].shape[:2] == img_array[0].shape[:2]:
                img_array[x] = cv2.resize(img_array[x], (0,0), None, scale, scale)
            else:
                img_array[x] = cv2.resize(img_array[x], (int(img_array[0].shape[1]), img_array[0].shape[0]), None, scale, scale)
            if len(img_array[x].shape) ==2: img_array[x] = cv2.cvtColor(img_array[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(img_array)
        ver = hor
    return ver


def is_white_pixel(pixel, threshold=190):
    """
    Determina si un píxel es blanco basado en un umbral.
    
    :param pixel: Un arreglo de numpy que representa un píxel (en formato BGR).
    :param threshold: Un valor de umbral para considerar un píxel como blanco (default 240).
    :return: True si el píxel es blanco, False de lo contrario.
    """
    return all(channel >= threshold for channel in pixel)


def is_gray_pixel(pixel, tolerance=55):
    """
    Determina si un píxel es gris y bajo un umbral.
    
    :param pixel: Un arreglo de numpy que representa un píxel (en formato BGR).
    :param threshold: Un valor de umbral para considerar un píxel como gris oscuro (default 128).
    :param tolerance: La tolerancia dentro de la cual los valores de los canales deben caer para considerarse gris (default 15).
    :return: True si el píxel es gris y oscuro, False de lo contrario.
    """
    # Convertir los valores de píxeles a int para evitar desbordamiento
    r, g, b = int(pixel[2]), int(pixel[1]), int(pixel[0])
    
    # Un píxel es gris si los valores de R, G y B son aproximadamente iguales
    # Y es "bajo" si esos valores son menores que el umbral
    return (abs(r - g) <= tolerance and
            abs(g - b) <= tolerance and
            abs(b - r) <= tolerance)


def get_top_contour(img, drawable_top_contour, img_width, img_height):
    bucket_top_contour = np.empty(img_width, dtype=object)

    for point in drawable_top_contour:
        if len(point) > 1:
            print("hi")

        x_pos = point[0][0]
        y_pos = point[0][1]
        
        if bucket_top_contour[x_pos] is None:
            bucket_top_contour[x_pos] = y_pos

        elif y_pos <  bucket_top_contour[x_pos]:
            bucket_top_contour[x_pos] = y_pos

    
    bucket_top_contour = [ img_height if y is None else y for y in bucket_top_contour ]

    drawable_top_contour = []

    for x, y in enumerate(bucket_top_contour):
        
        while is_white_pixel(img[y + 8, x]):
            y = y + 1

        drawable_top_contour.append(np.array([[x, y]], dtype=np.int32))
    
    return bucket_top_contour, drawable_top_contour


def display_results(result):
    image_ids = random.sample(indexes[result], 30)

    same_class_images = []
    current_row = []

    for image_index in image_ids:

        img_path = f"CIELOS_900/IMAGE{image_index}.jpg"

        current_row.append(cv2.imread(img_path))
        
        if len(current_row) == 5:
            same_class_images.append(current_row)
            current_row = []

        
        if len(same_class_images) == 2:
            break


    img_stack = stack_images(0.4, same_class_images)

    cv2.imshow("Result:", img_stack)



def predict_sky(img, img_contour, original):
    global start, station, difference

    contours, hierarchy = cv2.findContours(img, cv2. RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    biggest_contour = None
    biggest_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)


        if area > biggest_area:
            biggest_area = area
            biggest_contour = contour

    img_height, img_width = img.shape

    bucket_top_contour, drawable_top_contour = get_top_contour(original, biggest_contour, img_width, img_height)

    pixels_count = 0
    cloud_pixels_count = 0

    for col in range(img_width):
        for row in range(img_height):
            if row > bucket_top_contour[col]:
                break

            pixels_count += 1

            if is_white_pixel(original[row, col], threshold=140) or is_gray_pixel(original[row, col]):
                cloud_pixels_count += 1


    white_pixel_ratio = (cloud_pixels_count / pixels_count) * 100

    print(f"Cloud pixel ratio: {white_pixel_ratio}")
    
    cv2.drawContours(img_contour, drawable_top_contour, -1, (255, 0, 255), 5)
    
    if white_pixel_ratio <= 30:
        return BRILLIANT

    elif white_pixel_ratio > 30 and white_pixel_ratio < 80:
        return PARTIALLY_CLOUDED
    
    else:
        return CLOUDY


def predict(image_path):
    img = cv2.imread(image_path)


    img_contour = img.copy()
    img_blur = cv2.GaussianBlur(img, (7, 7), 1)
    img_gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)

    t_1 = 78
    t_2 = 15

    img_canny = cv2.Canny(img_gray, t_1, t_2)

    kernel = np.ones((5, 5))
    # dilated image
    img_dil = cv2.dilate(img_canny, kernel, iterations=1)

    return predict_sky(img_dil, img_contour, img)


Tk().withdraw()
img_path = askopenfilename()

original_img = cv2.imread(img_path)

cv2.imshow("Imagen origen", original_img)

result = predict(img_path)
print(f"Predicción: {result}")

display_results(result)


while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Presiona 'q' para salir
        break

cv2.destroyAllWindows()