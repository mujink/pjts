import os
import sys
import matplotlib.pyplot as plt
import xml.etree.ElementTree as Et
from xml.etree.ElementTree import Element, ElementTree


import numpy as np
import cv2
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

def drow_rectangle(img, coor):
    return img

IMAGE_FOLDER = "D:\python\pjt_odo\ImagesPart2"
ANNOTATIONS_FOLDER = "D:\python\pjt_odo/train_gt_t13"

ann_root, ann_dir, ann_files = next(os.walk(ANNOTATIONS_FOLDER))
img_root, amg_dir, img_files = next(os.walk(IMAGE_FOLDER))
for text_file in ann_files:

    img_name = img_files[img_files.index(".".join([text_file.split(".")[0], "jpg"]))]
    image = Image.open(os.path.join(img_root, img_name)).convert("RGB")
    draw = ImageDraw.Draw(image)

    text = open(os.path.join(ann_root, text_file), "r", encoding="utf-8")
    text = text.readlines()
    fonts = 'D:\python\pjt_odo/7/fonts/NanumSquareR.ttf'
    img = np.array(image)
    mask = np.zeros(img.shape[:2], np.uint8)
    for index ,line in enumerate(text):
        value = line.strip().split(",")

        left_up_x = float(value[0]) 
        left_up_y = float(value[1])
        right_up_x = float(value[2])
        right_up_y = float(value[3])
        right_donw_x = float(value[4])
        right_donw_y = float(value[5])
        left_donw_x = float(value[6])
        left_donw_y = float(value[7])
        typ = str(value[8])
        name = str(value[9])
        if typ == "Korean" and (name is not "###"):
            draw.line((left_up_x,left_up_y, right_up_x,right_up_y), fill=255, width= 15)
            draw.line((right_up_x,right_up_y, right_donw_x,right_donw_y), fill=255, width= 15)
            draw.line((right_donw_x,right_donw_y, left_donw_x,left_donw_y), fill=255, width= 15)
            draw.line((left_donw_x,left_donw_y, left_up_x,left_up_y), fill=255, width= 15)

            text_font = ImageFont.truetype(fonts, 1)
            draw.text(((left_up_x), (left_up_y - 150)), name, fill=(255), font=text_font)

            img = cv2.imread(os.path.join(img_root, img_name))
            pts = np.array([[left_donw_x,left_donw_y],
                            [right_donw_x,right_donw_y],
                            [right_up_x,right_up_y],
                            [left_up_x,left_up_y]])

            ## (2) make mask
            pts = pts.astype(np.int32)
            cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)

            # (3) do bit-op
            dst = cv2.bitwise_and(img, img, mask=mask)

    # cv2.imwrite("./mask/{}_{}_croped.png".format(img_name, index), croped)
    cv2.imwrite("./mask/{}_{}_mask.png".format(img_name, index), mask)
    cv2.imwrite("./mask/{}_{}_dst.png".format(img_name, index), dst)

    image.save('./mask/{}.png'.format(img_name),"PNG")
