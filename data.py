import numpy as np
from PIL import Image

red_dot = np.zeros(shape=(3,3,3),dtype=np.uint8)
red_dot[1][1][0]=255
green_dot = np.zeros(shape=(3,3,3),dtype=np.uint8)
green_dot[1][1][1]=255
blue_dot = np.zeros(shape=(3,3,3),dtype=np.uint8)
blue_dot[1][1][2]=255

red_dot = np.zeros(shape=(3,3,3),dtype=np.uint8)
red_dot[0][1][1]=255
green_dot = np.zeros(shape=(3,3,3),dtype=np.uint8)
green_dot[1][1][1]=255
blue_dot = np.zeros(shape=(3,3,3),dtype=np.uint8)
blue_dot[2][1][1]=255

r = Image.fromarray(red_dot)
g = Image.fromarray(green_dot)
b = Image.fromarray(blue_dot)

r.save("red_dot_n.png")
g.save("green_dot_n.png")
b.save("blue_dot_n.png")