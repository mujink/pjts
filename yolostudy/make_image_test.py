#!/usr/bin/env python

import glob
import io
import os
from PIL import Image, ImageFont, ImageDraw
import numpy as np
from sklearn.utils import shuffle
from config import cfg
SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))

# Default data paths.
DEFAULT_LABEL_FILE = os.path.join(SCRIPT_PATH, 'D:\python\pjt_odo/2주차\labels/2368-common-hangul.txt')
DEFAULT_FONTS_DIR = os.path.join(SCRIPT_PATH, 'D:\python\pjt_odo/2주차/fonts')
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_PATH, 'D:\python\pjt_odo/2주차/Test_image')

# Number of random distortion images to generate per font and character.
COUNT = 40000

# Width and height of the resulting image.
IMAGE_WIDTH = cfg.TRAIN.INPUT_SIZE[0]
IMAGE_HEIGHT = cfg.TRAIN.INPUT_SIZE[1]

Font_size = np.arange(36,100,2,np.int)
with io.open(DEFAULT_LABEL_FILE, 'r', encoding='utf-8') as f:
    labels = f.read().splitlines()

image_dir = os.path.join(DEFAULT_OUTPUT_DIR, 'images')
if not os.path.exists(image_dir):
    os.makedirs(os.path.join(image_dir))

fonts = glob.glob(os.path.join(DEFAULT_FONTS_DIR, '*.ttf'))

prev_count = 0
total_count =  10784
import tensorflow as tf
# Print image count roughly every 5000 images.
with tf.device("/gpu:0"):
    for i in range(COUNT):
        total_count += 1
        if total_count - prev_count > 5000:
            prev_count = total_count
            print('{} images generated...'.format(total_count))

        labels_txt = io.open(os.path.join(DEFAULT_OUTPUT_DIR, '{}.text'.format(total_count)), 'w',
                    encoding='utf-8')

        image = Image.new('L', (IMAGE_WIDTH, IMAGE_HEIGHT), color=0)
        font_size = shuffle(Font_size)[1]
        drawing = ImageDraw.Draw(image)
        
        font_heigh = font_size*2 + font_size*0.2
        font_width = font_size*2 + font_size*0.2

        text_heigh = np.arange(font_size + 20,IMAGE_HEIGHT+IMAGE_HEIGHT*0.85,font_heigh)
        text_width = np.arange(font_size + 20,IMAGE_WIDTH+IMAGE_WIDTH*0.85,font_width)
        
        for y_index, y1 in enumerate(text_heigh):
            for x_index, x1 in enumerate(text_width):
                labels = shuffle(labels)
                character = labels[(x_index+1)*(y_index+1)]
                try:
                    font = shuffle(fonts)[1]
                    text_font = ImageFont.truetype(font, font_size)
                    w, h = drawing.textsize(character, font=text_font)
                    drawing.text((((x1)/2), ((y1)/2)),
                                character,
                                fill=(255),
                                font=text_font
                                )
                    labels_txt.write(u'{},{},{},{},{}\n'.format(int((x1)/2), int((y1)/2), int(((x1)/2)+w), int(((y1)/2)+h), character))
                except:
                    continue
            
        file_string = '{}.jpeg'.format(total_count)
        file_path = os.path.join(image_dir, file_string)
        image.save(file_path, 'JPEG')
        labels_txt.close()      