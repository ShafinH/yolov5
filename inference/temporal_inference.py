import numpy as np
from PIL import Image

dir = '../drive/MyDrive/Research/VRL/val_data_30_frames/'

valfull_auspec = open(dir + 'valfull_auspec_cvat_frames_30.txt', 'r')
frames = valfull_auspec.readlines()
for frame in frames:
  img_array = np.load(dir + 'valfull_auspec_30/' + frame.strip() + '.npy')
  im = Image.fromarray(img_array)
  im.save(dir + 'imgs/' + frame.strip() + ".jpg")

!python detect.py --weights ../drive/MyDrive/Research/VRL/best.pt --source '../drive/MyDrive/Research/VRL/val_data_30_frames/imgs/'