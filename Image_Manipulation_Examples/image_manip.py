from PIL import Image
import numpy as np

img = Image.open('ex_image1.jpg')
print(img)
print(img.size)
print(img.mode)
print(img.width)
print(img.height)

pixels = img.load()

for x in range(img.width):
    for y in range(img.height):
        r, g, b = pixels[x, y]
        pixels[x, y] = (r//2, g//2, b)

for i in range(img.width):
    for j in range(img.height):
        if (i >= 600 & i <= 700) & (j >= 550 & j <= 620):
            r, g, b = pixels[i, j]
            pixels[i, j] = (255, 0, 0)

np_image = np.array(img)
print(len(np_image))
# for i in range(np_image):


img.show()