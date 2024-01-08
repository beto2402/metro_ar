import cv2
import numpy as np
from m_predecir import predecir

cap = cv2.VideoCapture(0)



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


def cambiar_permitir_prediccion(_enabled):
    global permitir_prediccion
    permitir_prediccion = _enabled

    cv2.setTrackbarPos("enabled", "Prediction", 1 if _enabled else 0)


def es_cuadrado(altura, ancho):
    suma = altura + ancho
    lado_corto = min(altura, ancho)

    return (lado_corto / suma) * 100 > 45


def obtener_contorno(img, img_contour, original):
    global start, estacion, permitir_prediccion

    contours, hierarchy = cv2.findContours(img, cv2. RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for contour in contours:
        area = cv2.contourArea(contour)

        # Discard very small figures
        if area > 1100:
            perimeter = cv2.arcLength(contour, True)

            # Busca el polígono que encierre el contorno dado. 
            # La última variable nos dice que debe ser un contorno cerrado
            approx_bounding = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

            # Verifica número de lados
            if len(approx_bounding) not in {4, 5}:
                continue


            # Obtiene el rectánculo que contiene los límites obtenidos
            x_, y_, w, h = cv2.boundingRect(approx_bounding)

            if not es_cuadrado(w, h):
                continue

            # Dibuja el cuadrado en la imagen que se muestra
            cv2.rectangle(img_contour, (x_, y_), (x_ + w, y_ + h), (0, 255, 0), 5)

            # Escribe el texto del resultado en la imagen
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

        img_canny = cv2.Canny(img_gray, 48, 15)

        kernel = np.ones((5, 5))

        # Imagen dilatada con kernel de unos
        img_dil = cv2.dilate(img_canny, kernel, iterations=1)

        cuadrado_recortado = obtener_contorno(img_dil, img_contour, img)


        if cuadrado_recortado is not None and permitir_prediccion:
            cambiar_permitir_prediccion(False)
            
            path_cuadrado_recortado = "cuadrado.jpg"
            cv2.imwrite(path_cuadrado_recortado, cuadrado_recortado)

            # Se hace la predicción y se asigna el valor
            estacion = predecir(path_cuadrado_recortado)            
                

        cv2.imshow("Result:", img_contour)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


main()