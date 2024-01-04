import cv2
import numpy as np
import time


frame_width = 640
frame_height = 480

cap = cv2.VideoCapture(0)
cap.set(3, frame_width)
cap.set(4, frame_height)

def empty(_):
    pass

cv2.namedWindow("Parameters")
cv2.resizeWindow("Parameters", 640, 240)
cv2.createTrackbar("Threshold1", "Parameters", 48, 255, empty)
cv2.createTrackbar("Threshold2", "Parameters", 15, 255, empty)


cv2.namedWindow("Prediction")
cv2.resizeWindow("Prediction", 640, 240)
cv2.createTrackbar("enabled", "Prediction", 1, 1, empty)



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

def prediction_enabled():
    return cv2.getTrackbarPos("enabled", "Prediction") == 1

def get_countours(img, img_contour, original, cropped):
    global start, station, difference

    contours, hierarchy = cv2.findContours(img, cv2. RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for contour in contours:
        area = cv2.contourArea(contour)

        if area > 1100 and area < 55000:
            #cv2.drawContours(img_contour, contour, -1, (255, 0, 255), 7)

            perimeter = cv2.arcLength(contour, True)

            # Find the bounding polygon for the given contour. 
            # The last var spacifies the contour must be closed
            approx_bounding = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

            if len(approx_bounding) < 4 or len(approx_bounding) > 5:
                continue

            if start == 0:
                start = time.time()

            else:
                difference = int(time.time() - start)


            # Get the bounding rectangle out of the given boundings
            x_, y_, w, h = cv2.boundingRect(approx_bounding)


            if difference != None and difference > 5 and prediction_enabled():
                return original[y_:y_+h, x_:x_+w]


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

    


while True:
    success, img = cap.read()

    img_contour = img.copy()
    img_cropped = None
    img_blur = cv2.GaussianBlur(img, (7, 7), 1)
    img_gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)

    t_1 = 48
    t_2 = 15 # 67?

    threshold_1 = cv2.getTrackbarPos("Threshold1", "Parameters")
    threshold_2 = cv2.getTrackbarPos("Threshold2", "Parameters")

    img_canny = cv2.Canny(img_gray, t_1, t_2)

    kernel = np.ones((5, 5))
    # dilated image
    img_dil = cv2.dilate(img_canny, kernel, iterations=1)

    cropped = get_countours(img_dil, img_contour, img, img_cropped)


    if cropped is not None and prediction_enabled:
        cv2.setTrackbarPos("enabled", "Prediction", 0)
        cv2.imwrite("cropped_image.jpg", cropped)

        # Here we will need to make the prediction and assign the value
        station = "wuuuuuuuuuuuuuuuuuuuu"


    img_stack = stack_images(0.8, ([img, img_gray, img_canny],
                                   [img_dil, img_dil, img_contour]))

    cv2.imshow("Result:", img_contour)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


