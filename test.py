from PIL import Image


img = Image.open("datasets/mios/ogs/isabelacatolica.png")

if img.mode in ('RGBA', 'LA'):
    # Create a new background image in white color
    background = Image.new(img.mode[:-1], img.size, (255, 255, 255))
    # Paste the original image onto the background
    background.paste(img, img.split()[-1])
    img = background

# Convert the image to RGB mode if it's not
if img.mode != 'RGB':
    img = img.convert('RGB')

img.save("test.jpg")
