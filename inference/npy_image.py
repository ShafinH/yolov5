import numpy as np
from PIL import Image


img_array = np.load('00002016080721161600_0013630.npy')

im = Image.fromarray(img_array)

im.save("test.jpg")
