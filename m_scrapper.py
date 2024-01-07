from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager # pip install webdriver-manager
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium import webdriver
import urllib.request
import cv2
import os


## Para correr el archivo es necesario crear las siguientes carpetas:
## - imagenes/o
## - imagenes/m



nombres_lineas = [
    "linea-1",
    "linea-2",
    "linea-3",
    "linea-4",
    "linea-5",
    "linea-6",
    "linea-7",
    "linea-8",
    "linea-9",
    "linea",
    "linea-b",
    "linea-12-2",
]

xpath_imagenes = "//img[contains(@class, 'fr-fic')]"
base_path = "imagenes/originales"

if not os.path.exists(base_path): 
           os.makedirs(base_path)

opts = Options()
opts.add_argument("user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.5938.152 Safari/537.36")

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=opts)


for nombre_linea in nombres_lineas:
    driver.get(f'https://www.metro.cdmx.gob.mx/la-red/{nombre_linea}')
    nombre_split = nombre_linea.split("-")

    id_linea = "a" if len(nombre_split) == 1 else nombre_split[1]

    estaciones = driver.find_elements(By.XPATH, xpath_imagenes)

    for i, estacion in enumerate(estaciones):
        url = estacion.get_attribute("src")
        nombre_imagen = f"{id_linea}.{i+1}.jpg"
        
        path_imagen_original = f"tmp_{nombre_imagen}"
        urllib.request.urlretrieve(url, path_imagen_original)

        img = cv2.imread(path_imagen_original, cv2.IMREAD_UNCHANGED)
    
        dim = (144, 144)

        trans_mask = img[:,:,3] == 0

        #replace areas of transparency with white and not transparent
        img[trans_mask] = [255, 255, 255, 255]

        #new image without alpha channel...
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        os.remove(path_imagen_original)

        cv2.imwrite(
              f"{base_path}/{nombre_imagen}",
              cv2.resize(img, dim, interpolation = cv2.INTER_AREA))

