from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager # pip install webdriver-manager
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium import webdriver
import urllib.request
from PIL import Image
import os
from preprocess import resize_and_save, save_png_as_jpg


la_red_dropdown_xpath = "((//a[@href='https://www.metro.cdmx.gob.mx/la-red'])[1]//following::ul[1]//child::a[contains(@href, 'www')])"
base_url = "https://www.metro.cdmx.gob.mx"

base_path = "datasets/mios"

if not os.path.exists(base_path): 
    os.makedirs(base_path)

opts = Options()
opts.add_argument("user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.5938.152 Safari/537.36")

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=opts)

driver.get(base_url)


lines_links_elements = [ele.get_attribute("href") for ele in driver.find_elements(By.XPATH, la_red_dropdown_xpath)]


for i, line_url in enumerate(lines_links_elements):
    # The first link goes to a map, not to a line
    if i == 0:
        continue
    
    driver.get(line_url)

    line_id = i

    if i == 10:
        line_id = "a"
    elif i == 11:
        line_id = "b"

    page_images_xpath = f"//img[contains(@src, 'linea{line_id}')]"

    stations_images_elements = driver.find_elements(By.XPATH, page_images_xpath)


    for j, station_element in enumerate(stations_images_elements):
        # The width of the icons is not higher than 100
        if int(station_element.get_attribute("width")) > 100:
            continue

        url = station_element.get_attribute("src")
        station_name = url.split("/")[-1].split(".")[0]
        ogs_path = f"{base_path}/ogs"
        og_img_path = f"{ogs_path}/{station_name}.png"

        if os.path.exists(og_img_path):
            continue

        
        if not os.path.exists(ogs_path): 
           os.makedirs(ogs_path)
        
        

        resizeds_path = f"{base_path}/resized"
        if not os.path.exists(resizeds_path):
           os.makedirs(resizeds_path)

        
        jpgs_ogs_path = f"{base_path}/jpgs"
        if not os.path.exists(jpgs_ogs_path):
           os.makedirs(jpgs_ogs_path)
        

        urllib.request.urlretrieve(url, og_img_path)

        jpg_og_img_path = f"{base_path}/jpgs/{station_name}.jpg"

        save_png_as_jpg(og_img_path, new_path)
        
        resized_img_path = f"{base_path}/resized/{station_name}.jpg"

        new_path = resize_and_save(jpg_og_img_path, resized_img_path)



