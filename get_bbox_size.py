import math
import os
import numpy as np
import pandas as pd
from PIL import Image


labels_df = pd.read_csv('labels.csv')
print(labels_df)
areaaaa = []
xxxxx = []
for ii, row in labels_df.iterrows():
    image_path = os.path.join('dataset', 'images', row['image_id'] + '.bmp')
    image = np.asarray(Image.open(image_path), dtype=np.uint8)
    h, w = image.shape[:2]
    #  print(h, w)
    dx = w * (row.xmax - row.xmin)
    dy = h * (row.ymax - row.ymin)
    area = dx * dy
    print(dx, dy, area)
    areaaaa.append(area)
    xxxxx.append(dx)
    xxxxx.append(dy)

avg_area = sum(areaaaa) / len(areaaaa)
print(avg_area)
print(math.sqrt(avg_area))
avg_area = sum(xxxxx) / len(xxxxx)
print(avg_area)
print(math.sqrt(avg_area))
