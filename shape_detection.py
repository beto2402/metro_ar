import cv2
import numpy as np
import time
import asyncio
from m_predecir import predecir


frame_width = 640
frame_height = 480

cap = cv2.VideoCapture(0)
cap.set(3, frame_width)
cap.set(4, frame_height)



permitir_prediccion = True

def cambiar_permitido(value):
    global start, permitir_prediccion, estacion
    
    if value == 1:
        start = 0
        estacion = "?"
    
    permitir_prediccion = value == 1


cv2.namedWindow("Prediction")
cv2.resizeWindow("Prediction", 640, 240)
cv2.createTrackbar("enabled", "Prediction", 1 if permitir_prediccion else 0, 1, cambiar_permitido)


estacion = "?"
diferencia = None
tiempo_de_reconocimiento = 3


def cambiar_permitir_prediccion(_enabled):
    global permitir_prediccion
    permitir_prediccion = _enabled

    cv2.setTrackbarPos("enabled", "Prediction", 1 if _enabled else 0)


def es_cuadrado(altura, ancho):
    suma = altura + ancho
    lado_corto = min(altura, ancho)

    return (lado_corto / suma) * 100 > 45


def obtener_contorno(img, img_contour, original):
    global start, estacion, diferencia, tiempo_de_reconocimiento, permitir_prediccion

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



            # Get the bounding rectangle out of the given boundings
            x_, y_, w, h = cv2.boundingRect(approx_bounding)

            if not es_cuadrado(w, h):
                continue


            # Draw the rectangle in the image
            cv2.rectangle(img_contour, (x_, y_), (x_ + w, y_ + h), (0, 255, 0), 5)

            
            cv2.putText(img_contour, f"Resultado: {estacion}",
                        (x_ + w + 20, y_ + 70), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                        (255, 0, 255), 2)
            

            if permitir_prediccion:
                return original[y_:y_+h, x_:x_+w]
            


def main():
    global estacion, permitir_prediccion

    while True:
        _, img = cap.read()

        img_contour = img.copy()
        img_blur = cv2.GaussianBlur(img, (7, 7), 1)
        img_gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)

        t_1 = 48
        t_2 = 15

        img_canny = cv2.Canny(img_gray, t_1, t_2)

        kernel = np.ones((5, 5))

        # Imagen dilatada con kernel de unos
        img_dil = cv2.dilate(img_canny, kernel, iterations=1)

        cropped = obtener_contorno(img_dil, img_contour, img)


        if cropped is not None and permitir_prediccion:
            cambiar_permitir_prediccion(False)
            
            cropped_path = "/tmp/cropped_image.jpg"
            cv2.imwrite(cropped_path, cropped)

            # Se hace la predicción y se asigna el valor
            estacion = predecir(cropped_path)
                

        cv2.imshow("Result:", img_contour)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


asyncio.run(main())