import cv2
import numpy as np
import time
import os
from tt_constants import NORMALIZED_GS_PATH


frame_width = 640
frame_height = 480

cap = cv2.VideoCapture(0)
cap.set(3, frame_width)
cap.set(4, frame_height)

def empty(_):
    pass

cv2.namedWindow("Parameters")
cv2.resizeWindow("Parameters", 640, 240)
cv2.createTrackbar("Threshold1", "Parameters", 255, 600, empty)
cv2.createTrackbar("Threshold2", "Parameters", 255, 600, empty)




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


start = 0
station = ""
difference = None


def get_countours(img, img_contour, original, cropped):
    global start, station, difference

    contours, hierarchy = cv2.findContours(img, cv2. RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for contour in contours:
        area = cv2.contourArea(contour)

        if area > 1000 and area < 10000:
            #cv2.drawContours(img_contour, contour, -1, (255, 0, 255), 7)

            perimeter = cv2.arcLength(contour, True)

            # Find the bounding polygon for the given contour. 
            # The last var spacifies the contour must be closed
            approx_bounding = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

            # if len(approx_bounding) < 4 or len(approx_bounding) > 5:
            #     continue


            # Get the bounding rectangle out of the given boundings
            x_, y_, w, h = cv2.boundingRect(approx_bounding)


            # Draw the rectangle in the image
            cv2.rectangle(img_contour, (x_, y_), (x_ + w, y_ + h), (0, 255, 0), 5)

            cv2.putText(img_contour, f"Points: {len(approx_bounding)}",
                        (x_ + w + 20, y_ + 20), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                        (0, 255, 0), 2)

            cv2.putText(img_contour, f"Area: {int(area)}",
                        (x_ + w + 20, y_ + 45), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                        (0, 255, 0), 2)

            cv2.putText(img_contour, f"Estación: {station}",
                        (x_ + w + 20, y_ + 70), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                        (0, 255, 0), 2)

img_path = f"{NORMALIZED_GS_PATH}/{sorted(os.listdir(NORMALIZED_GS_PATH))[2]}"
img = cv2.imread(img_path)


while True:
    img_contour = img.copy()
    img_cropped = None
    img_blur = cv2.GaussianBlur(img, (7, 7), 1)

    ## 400 - 81
    ## 304 - 96
    ## 376 - 102

    t_1 = 376
    t_2 = 102 # 67?

    t_1 = cv2.getTrackbarPos("Threshold1", "Parameters")
    t_2 = cv2.getTrackbarPos("Threshold2", "Parameters")

    img_canny = cv2.Canny(img, t_1, t_2)

    kernel = np.ones((3, 3))
    # dilated image
    img_dil = cv2.dilate(img_canny, kernel, iterations=1)

    cropped = get_countours(img_dil, img_contour, img, img_cropped)



    img_stack = stack_images(0.8, ([img, img_canny],
                                   [img_dil, img_contour]))

    cv2.imshow("Result:", img_stack)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break