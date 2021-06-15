#!/usr/bin/env python

import glob
import io
import os
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
import glob
import cv2

def image_preprocess(image, target_size, gt_boxes=None):
    
    ih, iw    = target_size
    h,  w, c  = image.shape

    scale = min(iw/w, ih/h)

    nw, nh  = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    image_paded = np.full(shape=[ih, iw, 3], fill_value=0.0)
    dw, dh = (iw - nw) // 2, (ih - nh) // 2

    image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized
    image_paded = image_paded / 255.

    return image_paded

# Default data paths.
DEFAULT_LABEL_FILE = os.path.join(SCRIPT_PATH, 'D:\python\pjt_odo/7\labels/5888-common-hangul.txt')
DEFAULT_FONTS_DIR = os.path.join(SCRIPT_PATH, 'D:\python\pjt_odo/7/fonts')
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_PATH, 'D:\python\pjt_odo/7/Test_image')
BACKGROUND_IMAGE = glob.glob(os.path.join("./img", '*.png'))
# Number of random distortion images to generate per font and character.
COUNT = 10000

# Width and height of the resulting image.
IMAGE_WIDTH = 416
# IMAGE_HEIGHT = int(IMAGE_WIDTH * 1.4142)
IMAGE_HEIGHT = 416

Font_size = np.arange(16,38,4,np.int)
with io.open(DEFAULT_LABEL_FILE, 'r', encoding='utf-8') as f:
    labels = f.read().splitlines()

image_dir = os.path.join(DEFAULT_OUTPUT_DIR, 'images')
if not os.path.exists(image_dir):
    os.makedirs(os.path.join(image_dir))

fonts = glob.glob(os.path.join(DEFAULT_FONTS_DIR, '*.ttf'))

prev_count = 0
total_count =  0
# Print image count roughly every 5000 images.
with tf.device("/gpu:0"):
    for i in range(COUNT):
        total_count += 1
        if total_count - prev_count > 5000:
            prev_count = total_count
            print('{} images generated...'.format(total_count))

        labels_txt = io.open(os.path.join(DEFAULT_OUTPUT_DIR, '{}.text'.format(total_count)), 'w',
                    encoding='utf-8')

        # image = Image.new('L', (IMAGE_WIDTH, IMAGE_HEIGHT), color=0)
        background = shuffle(BACKGROUND_IMAGE)[1]
        image = Image.open(background).convert("RGB")
        font_size = shuffle(Font_size)[1]
        image = image_preprocess(np.copy(image), [IMAGE_HEIGHT, IMAGE_WIDTH])
        image = Image.fromarray((image*255).astype(np.uint8))

        drawing = ImageDraw.Draw(image)

        font_heigh = font_size*2 + font_size*0.2
        font_width = font_size*8 + font_size*0.2

        text_heigh_all = np.arange(font_size + 20,IMAGE_HEIGHT+IMAGE_HEIGHT*0.85,font_heigh)
        text_width_all = np.arange(font_size + 20,IMAGE_WIDTH+IMAGE_WIDTH*0.85,font_width)
        random_h = int(np.random.randint(len(text_heigh_all)//4,len(text_heigh_all)-1,1))

        text_heigh = sorted(shuffle(text_heigh_all)[:random_h])
        for y_index, y1 in enumerate(text_heigh):
            random_w = int(np.random.randint(len(text_width_all)//3,len(text_width_all)-1,1))
            text_width = sorted(shuffle(text_width_all)[0:random_w])
            for x_index, x1 in enumerate(text_width):
                labels = shuffle(labels)
                character = labels[(x_index+1)*(y_index+1)]
                font = shuffle(fonts)[1]
                text_font = ImageFont.truetype(font, font_size)
                w, h = drawing.textsize(character, font=text_font)
                if ((x1)/2)+w < IMAGE_WIDTH and ((y1)/2)+h < IMAGE_HEIGHT:
                    r, g, b = np.random.randint(0,256,3)
                    drawing.text(((x1/2), (y1/2)),
                                character,
                                fill=(r,g,b),
                                font=text_font,
                                )
                    labels_txt.write(u'{},{},{},{},{}\n'.format(int((x1)/2), int((y1)/2), int(((x1)/2)+w), int(((y1)/2)+h), character))

        file_string = '{}.jpeg'.format(total_count)
        file_path = os.path.join(image_dir, file_string)
        image.save(file_path, 'JPEG')
        labels_txt.close()