import cv2
import numpy as np
import time
from yolo_predict import YoloPredict
import asyncio
from utils import stack_images


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


prediction_enabled = True

def change_enabled(value):
    global start, prediction_enabled, station
    
    if value == 1:
        start = 0
        station = "?"
    
    prediction_enabled = value == 1


cv2.namedWindow("Prediction")
cv2.resizeWindow("Prediction", 640, 240)
cv2.createTrackbar("enabled", "Prediction", 1 if prediction_enabled else 0, 1, change_enabled)


start = 0
station = "?"
time_diff = None
recognition_time = 3


def set_prediction_enabled(_enabled):
    global prediction_enabled
    prediction_enabled = _enabled

    cv2.setTrackbarPos("enabled", "Prediction", 0)


def is_squareish(height, width):
    max_measurement = max(height, width)
    min_measurement = min(height, width)

    percentual_diff = (min_measurement / max_measurement) * 100
    
    return percentual_diff > 85


def get_countours(img, img_contour, original):
    global start, station, time_diff, recognition_time, prediction_enabled

    contours, hierarchy = cv2.findContours(img, cv2. RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for contour in contours:
        area = cv2.contourArea(contour)

        # Discard very small figures
        if area > 1100:
            #cv2.drawContours(img_contour, contour, -1, (255, 0, 255), 7)

            perimeter = cv2.arcLength(contour, True)

            # Find the bounding polygon for the given contour. 
            # The last var spacifies the contour must be closed
            approx_bounding = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

            #Check number of sides
            if len(approx_bounding) not in {4, 5}:
                continue

            if start == 0:
                start = time.time()

            else:
                time_diff = int(time.time() - start)


            # Get the bounding rectangle out of the given boundings
            x_, y_, w, h = cv2.boundingRect(approx_bounding)

            if not is_squareish(w, h):
                continue


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
            

            if time_diff != None and time_diff > recognition_time and prediction_enabled:
                return original[y_:y_+h, x_:x_+w]
            

async def set_station_name(_yolo_predict, img_path):
    global station
    station = await _yolo_predict.predict(img_path)
    print(station)


background_tasks = set()

async def main():
    global station, prediction_enabled
    yolo_predict = YoloPredict(grayscale=True)

    while True:
        success, img = cap.read()

        img_contour = img.copy()
        img_cropped = None
        img_blur = cv2.GaussianBlur(img, (7, 7), 1)
        img_gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)

        t_1 = 48
        t_2 = 15

        # t_1 = cv2.getTrackbarPos("Threshold1", "Parameters")
        # t_2 = cv2.getTrackbarPos("Threshold2", "Parameters")

        img_canny = cv2.Canny(img_gray, t_1, t_2)

        kernel = np.ones((5, 5))
        # dilated image
        img_dil = cv2.dilate(img_canny, kernel, iterations=1)

        cropped = get_countours(img_dil, img_contour, img)


        if cropped is not None and prediction_enabled:
            set_prediction_enabled(False)
            
            cropped_path = "/tmp/cropped_image.jpg"
            cv2.imwrite(cropped_path, cropped)

            # Here we will need to make the prediction and assign the value
            task = asyncio.create_task(
                set_station_name(yolo_predict, cropped_path)
                )
            
            await task

            background_tasks.add(task)
            task.add_done_callback(background_tasks.discard)

        # if len(background_tasks) > 0:
        #     cv2.namedWindow("Prediction")
        #     cv2.resizeWindow("Prediction", 640, 240)
        #     cv2.createTrackbar("enabled", "Prediction", 1 if prediction_enabled else 0, 1, change_enabled)


        img_stack = stack_images(0.8, ([img_canny, img_contour], [img_canny, img_contour]))

        cv2.imshow("Result:", img_contour)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


asyncio.run(main())