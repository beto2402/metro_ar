import cv2
import numpy as np
import time
import asyncio
from yolo_predict import YoloPredict


frame_width = 640
frame_height = 480


class MetroPredictor(YoloPredict):
    cap = cv2.VideoCapture(0)
    prediction_enabled = True
    start = 0
    time_diff = None
    recognition_time = None
    t_1 = None
    t_2 = None
    cropped = None
    image_contour = None
    station = ""
    cropped_path = ""
    background_tasks = set()
    img = None
    image_contour = None

    

    @property
    def time_completed(self):
        return self.time_diff != None and self.time_diff > self.recognition_time
    

    @property
    def dilated_image(self):
        img_blur = cv2.GaussianBlur(self.img, (7, 7), 1)
        img_gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
        

        img_canny = cv2.Canny(img_gray, self.t_1, self.t_2)
        
        # Kernel used to enhance contours
        # The size of the kernel affects the extent of dilation. 
        # A larger kernel will result in thicker edges.
        kernel = np.ones((5, 5))
        
        # dilated image
        return cv2.dilate(img_canny, kernel, iterations=1)
        
    
    def __init__(self, recognition_time=3, t_1=48, t_2=15, grayscale=True) -> None:
        super().__init__(grayscale=grayscale)

        self.cap.set(3, frame_width)
        self.cap.set(4, frame_height)
        self.recognition_time = recognition_time
        self.t_1 = t_1
        self.t_2 = t_2

        cv2.namedWindow("Prediction")
        cv2.resizeWindow("Prediction", 640, 240)
        cv2.createTrackbar("enabled", "Prediction", 1 if self.prediction_enabled else 0, 1, self.change_enabled)

    
    def change_enabled(self, value):
        if value == 1:
            self.start = 0
            self.station = "?"
        
        self.prediction_enabled = value == 1


    def reset_image_contour(self):
        self.image_contour = self.img.copy()

    
    def set_prediction_enabled(self, value):
        cv2.setTrackbarPos("enabled", "Prediction", 0 if not value else 1)

        self.prediction_enabled = value
    

    def is_squareish(self, height, width):
        max_measurement = max(height, width)
        min_measurement = min(height, width)

        percentual_diff = (min_measurement / max_measurement) * 100
        
        return percentual_diff > 85
    

    def set_time_diff(self):
        if self.start == 0:
            self.start = time.time()
    
        self.time_diff = int(time.time() - self.start)
    

    def get_bounding_rectangle(self, contour):
        area = cv2.contourArea(contour)

        # Discard very small figures
        if area > 1100:
            perimeter = cv2.arcLength(contour, True)

            # Find the bounding polygon for the given contour. 
            # The last var spacifies the contour must be closed
            approx_bounding = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

            #Check number of sides
            if len(approx_bounding) not in {4, 5}:
                return None


            self.set_time_diff()


            # Get the bounding rectangle out of the given boundings
            return cv2.boundingRect(approx_bounding)


    def get_cropped_square(self):
        contours, hierarchy = cv2.findContours(self.dilated_image, cv2. RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        for i, contour in enumerate(contours):

            bounding_rectangle = self.get_bounding_rectangle(contour)

            if bounding_rectangle == None:
                continue


            x_, y_, w, h = bounding_rectangle

            if not self.is_squareish(w, h):
                continue

            # Draw the rectangle in the image
            cv2.rectangle(self.image_contour, (x_, y_), (x_ + w, y_ + h), (0, 255, 0), 5)


            cv2.putText(self.image_contour, f"Estación: {self.station}",
                        (x_ + w + 20, y_ + 70), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                        (0, 255, 0), 2)
            

            if self.time_completed and self.prediction_enabled:
                return self.img[y_:y_+h, x_:x_+w]
            

    async def set_station_name(self):
        self.station = await self.predict(self.cropped_path)


    async def start_camera(self):
        while True:
            _, img = self.cap.read()
            self.img = img

            self.image_contour = img.copy()


            cropped = self.get_cropped_square()

            if cropped is not None and self.prediction_enabled:
                self.cropped = cropped
                self.set_prediction_enabled(False)

                self.cropped_path = f"/tmp/cropped_image_{time.time()}.jpg"
                cv2.imwrite(self.cropped_path, cropped)

                # Here we will need to make the prediction and assign the value
                task = asyncio.create_task(
                    self.set_station_name()
                    )
                
                await task

                self.background_tasks.add(task)
                task.add_done_callback(self.background_tasks.discard)


            cv2.imshow("Result:", self.image_contour)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


metro_predictor = MetroPredictor(grayscale=False)

asyncio.run(metro_predictor.start_camera())




