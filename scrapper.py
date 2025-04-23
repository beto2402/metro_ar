from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
# pip install webdriver-manager
from webdriver_manager.chrome import ChromeDriverManager
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium import webdriver
import urllib.request
from PIL import Image
import os
from preprocess import resize_and_save, save_png_as_jpg

import constants


def initialize_folders():
    for folder in constants.REQUIRED_PATHS:
        if not os.path.exists(folder):
            os.makedirs(folder)


def build_driver():
    browser_options = Options()
    browser_options.add_argument(
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_2) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.3 Safari/605.1.15"
    )

    return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=browser_options)


def find_elements_by_xpath(xpath):
    return driver.find_elements(By.XPATH, xpath)


def get_lines_urls_info():
    lines_dropdown_xpath = f"(//a[@href='{constants.BASE_URL}/la-red'])[1]"
    metro_lines_dropdown_elements_xpath = f"({lines_dropdown_xpath}//following::ul[1]//child::a[contains(@href, 'www')])"

    driver.get(constants.BASE_URL)
    lines_dropdown = find_elements_by_xpath(lines_dropdown_xpath)[0]
    lines_dropdown.click()

    target_urls_info = []

    for dropdown_element in find_elements_by_xpath(metro_lines_dropdown_elements_xpath):
        target_urls_info.append({
            "target_url": dropdown_element.get_attribute("href"),
            "name": dropdown_element.text,
        })

    return target_urls_info


def save_from_url(url, path):
    if os.path.exists(path):
        return

    urllib.request.urlretrieve(url, path)


def handle_metadata():
    service_hours_info_xpath = "//*[contains(text(), 'Días Laborales')][1]"
    free_access_info_xpath = "//*[contains(text(), 'Adultos mayores')][1]"
    cost_info_xpath = "//*[contains(text(), '$')][1]"

    metadata = {
        "service_hours_info": service_hours_info_xpath,
        "free_access_info": free_access_info_xpath,
        "cost_info": cost_info_xpath,
    }

    # Get metadata elements and save them
    for metadata_key in metadata.keys():
        content = find_elements_by_xpath(metadata[metadata_key])[
            0].text.strip()
        with open(f"{constants.GENERAL_INFORMATION_PATH}/{metadata_key}.txt", "w") as file:
            file.write(content)

    # Get integrated mobility map element
    integrated_mobility_map_xpath = "//img[contains(@src, 'MAPA_MI')][1]"
    image = find_elements_by_xpath(integrated_mobility_map_xpath)[0]
    url = image.get_attribute("src")

    # Save image from URL in current format
    save_from_url(
        url, f"{constants.GENERAL_INFORMATION_PATH}/integrated_mobility_map.png")


def handle_line_url_info(url_info):
    # Format is "Linea ID", so we only keep what's after the space
    line_id = url_info["name"].split(" ")[1].strip().lower()

    page_images_xpath = f"//img[contains(@src, 'linea{line_id}')]"

    print(f"-----PROCESSING LINE '{line_id}'-----")
    for station_element in find_elements_by_xpath(page_images_xpath):
        # The width of the icons is not higher than 100
        if int(station_element.get_attribute("width")) > 100:
            continue

        url = station_element.get_attribute("src")
        station_name = url.split("/")[-1].split(".")[0]

        print(f"Saving station '{station_name}'")

        og_img_path = f"{constants.ORIGINAL_IMAGES_PATH}/{station_name}.png"

        save_from_url(url, og_img_path)

        jpg_og_img_path = f"{constants.BASE_PATH}/jpgs/{station_name}.jpg"

        save_png_as_jpg(og_img_path, jpg_og_img_path)

        resized_img_path = f"{constants.BASE_PATH}/resized/{station_name}.jpg"

        resize_and_save(jpg_og_img_path, resized_img_path)


initialize_folders()
driver = build_driver()

target_urls_info = get_lines_urls_info()


for url_info in target_urls_info:
    driver.get(url_info["target_url"])

    # Handle Map differently
    if "Mapa" in url_info["name"]:
        handle_metadata()
    elif "Línea" in url_info["name"]:
        handle_line_url_info(url_info)
