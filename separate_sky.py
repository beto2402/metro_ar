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


def get_top_contour(drawable_top_contour, img_width, img_height):
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

    drawable_top_contour = [np.array([[x, y]], dtype=np.int32) for x, y in enumerate(bucket_top_contour)]
    
    return bucket_top_contour, drawable_top_contour

def prediction_enabled():
    return cv2.getTrackbarPos("enabled", "Prediction") == 1

def get_countours(img, img_contour, original, cropped):
    global start, station, difference

    contours, hierarchy = cv2.findContours(img, cv2. RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    biggest_contour = None
    biggest_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)


        if area > biggest_area:
            biggest_area = area
            biggest_contour = contour


    bucket_top_contour, drawable_top_contour = get_top_contour(biggest_contour, img.shape[1], img.shape[0])
    
    cv2.drawContours(img_contour, drawable_top_contour, -1, (255, 0, 255), 7)
    


    #[y, x]
    perimeter = cv2.arcLength(contour, True)

    # Find the bounding polygon for the given contour. 
    # The last var spacifies the contour must be closed
    approx_bounding = cv2.approxPolyDP(contour, 0.02 * perimeter, True)





while True:
    img = cv2.imread("CIELOS_900/IMAGE536.jpg")

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


    img_stack = stack_images(0.8, ([img_gray, img_canny, img_contour],
                                   [img_dil, img_dil, img_contour]))

    cv2.imshow("Result:", img_stack)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break