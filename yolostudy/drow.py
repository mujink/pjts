import os
import sys
import matplotlib.pyplot as plt
import xml.etree.ElementTree as Et
from xml.etree.ElementTree import Element, ElementTree

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

dataset_path = "./raccoon/"

IMAGE_FOLDER = "image"
ANNOTATIONS_FOLDER = "xml"
print(dataset_path)
print(os.path.join(dataset_path, ANNOTATIONS_FOLDER))

ann_root, ann_dir, ann_files = next(os.walk(os.path.join(dataset_path, ANNOTATIONS_FOLDER)))
img_root, amg_dir, img_files = next(os.walk(os.path.join(dataset_path, IMAGE_FOLDER)))

for xml_file in ann_files:

    # XML파일와 이미지파일은 이름이 같으므로, 확장자만 맞춰서 찾습니다.
    img_name = img_files[img_files.index(".".join([xml_file.split(".")[0], "jpg"]))]

    image = Image.open(os.path.join(img_root, img_name)).convert("RGB")
    draw = ImageDraw.Draw(image)

    xml = open(os.path.join(ann_root, xml_file), "r")
    tree = Et.parse(xml)
    root = tree.getroot()

    size = root.find("size")

    width = size.find("width").text
    height = size.find("height").text
    channels = size.find("depth").text

    objects = root.findall("object")

    for _object in objects:
        name = _object.find("name").text
        bndbox = _object.find("bndbox")
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)

        # Box를 그릴 때, 왼쪽 상단 점과, 오른쪽 하단 점의 좌표를 입력으로 주면 됩니다.
        draw.rectangle(((xmin, ymin), (xmax, ymax)), outline="red")
        draw.text((xmin, ymin), name)

    plt.figure(figsize=(25,20))
    plt.imshow(image)
    plt.show()
    plt.close()