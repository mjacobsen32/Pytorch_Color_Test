import numpy as np
from PIL import Image

'''
    Manually creating and saving dot images as pngs
'''

red_dot = np.zeros(shape=(3,3,3),dtype=np.uint8)
red_dot[1][1][0]=255
green_dot = np.zeros(shape=(3,3,3),dtype=np.uint8)
green_dot[1][1][1]=255
blue_dot = np.zeros(shape=(3,3,3),dtype=np.uint8)
blue_dot[1][1][2]=255

r = Image.fromarray(red_dot)
g = Image.fromarray(green_dot)
b = Image.fromarray(blue_dot)

r.save("red_dot.png")
g.save("green_dot.png")
b.save("blue_dot.png")