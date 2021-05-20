
from Coordinate import Text_Coordinate
import os
import io
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
# path = "D:\python\pjt_odo\pdf2_img"
# path = "D:\python\pjt_odo/test"
# label_path = "D:\python\pjt_odo/test_label"
# file = os.listdir(path)




# for j, f in enumerate(file):  
#     fpath = path + "/" + f
#     labels_txt = io.open(os.path.join(label_path, '{}.text'.format(f)), 'w',
#                          encoding='utf-8')
#     image = cv2.imread(fpath)
#     image = cv2.resize(image,(int(image.shape[0]/1.5),int(image.shape[1]/1.5)))

#     a = Text_Coordinate(image)
#     b = a.bbox()
#     print(f)
#     for i, word in enumerate(b):
#         line = i
#         for j in range(len(word)):
#             str_index = j
#             if b[line,str_index,0] == 1:
#                 xmin = b[line,str_index,1]
#                 ymin = b[line,str_index,2]
#                 xmax = b[line,str_index,3]
#                 ymax = b[line,str_index,4]
#                 if xmax-xmin <6:
#                     continue
#                 elif ymax-ymin <6:
#                     continue
#                 # green_color = (0,255,0)
#                 labels_txt.write(u'{},  {}, {}, {}, {}\n'.format(1, xmin, ymin, xmax, ymax))
#                 # image = cv2.rectangle(image, (xmin, ymin),(xmax,ymax),green_color,1)
#     labels_txt.close()
    # cv2.imshow("detect_Line : {}, Index_Sting : {}".format(line+1,str_index+1), image)
    # cv2.waitKey(0)
    # cv2.destroyWindow("detect")


path = ".\Test_image\images"
label_path = ".\Test_image"
file = os.listdir(path)

for j, f in enumerate(file):  
    fpath = path + "/" + "{}.jpeg".format(j+1)
    labels_txt = open(os.path.join(label_path,'{}.text'.format(j+1)), 'r',
                         encoding='utf-8')
    # print(fpath)
    # print(labels_txt)
    image = cv2.imread(fpath)
    # cv2.imshow("image",image)
    # cv2.waitKey(0)
    # cv2.destroyWindow("detect")
    # a = Text_Coordinate(image)
    # b = a.bbox()
    for i, word in enumerate(labels_txt):
        word = word.split(",")
        xmin = int(word[0])
        ymin = int(word[1])
        xmax = int(word[2])
        ymax = int(word[3])
        green_color = (0,255,0)
        image = cv2.rectangle(image, (xmin, ymin),(xmax,ymax),green_color,2)
    
    image = cv2.resize(image, (int(image.shape[0]*0.5), int(image.shape[1]*1)))
    cv2.imshow("image",image)
    cv2.waitKey(0)
    cv2.destroyWindow("detect")