
from Coordinate import Text_Coordinate
import os
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


    
def out_text(path):
    image_list = []
    for j, f in enumerate(os.listdir(path)):
        fpath = path + "/" + f

        image = cv2.imread(fpath)

        a = Text_Coordinate(image)
        b = a.bbox()
        images = []
        for i, word in enumerate(b):
            line = i
            for j in range(len(word)):
                str_index = j
                if b[line,str_index,0] == 1:
                    xmin = b[line,str_index,1]
                    ymin = b[line,str_index,2]
                    xmax = b[line,str_index,3]
                    ymax = b[line,str_index,4]
                    if xmax-xmin <6:
                        continue
                    elif ymax-ymin <6:
                        continue
                    green_color = (0,255,0)
                    # labels_txt.write(u'{},  {}, {}, {}, {}\n'.format(1, xmin, ymin, xmax, ymax))
                    # image = cv2.rectangle(image, (xmin, ymin),(xmax,ymax),green_color,2)
                    images.append(image[ymin:ymax+1,xmin:xmax+1])
        image_list.append(images)
    return image_list
