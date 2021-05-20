#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 17:43:39 2019
@author: deniz
"""

import json
import os
import math
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns; sns.set()  # for plot styling
import glob
import xml.etree.ElementTree as Et

# os.chdir("/media/deniz/02B89600B895F301/BBD100K")
train_path = "./raccoon/xml"
xml = glob.glob(os.path.join(train_path, '*.xml'))
w,h = [] , []
for i in range(len(xml)):
    print(i+1)
    with open(xml[i], "r") as fd:
        tree = Et.parse(fd)
        root = tree.getroot()
        objects = root.findall("object")
        for line in objects:
            box = line.find("bndbox")
            x1 = float(box.find("xmin").text)
            y1 = float(box.find("ymin").text)
            x2 = float(box.find("xmax").text)
            y2 = float(box.find("ymax").text)
            width = abs(x1-x2)
            height = abs(y1-y2)
            print(width, height)
            w.append(width)
            h.append(height)         


w=np.asarray(w)
h=np.asarray(h)

x=[w,h]
x=np.asarray(x)
x=x.transpose()
##########################################   K- Means
##########################################

from sklearn.cluster import KMeans
n_clusters=9
kmeans3 = KMeans(n_clusters=n_clusters)
kmeans3.fit(x)
y_kmeans3 = kmeans3.predict(x)

##########################################
centers3 = kmeans3.cluster_centers_

yolo_anchor_average=[]
for ind in range (n_clusters):
    yolo_anchor_average.append(np.mean(x[y_kmeans3==ind],axis=0))

yolo_anchor_average=np.array(yolo_anchor_average)

plt.scatter(x[:, 0], x[:, 1], c=y_kmeans3, s=2, cmap='viridis')
plt.scatter(yolo_anchor_average[:, 0], yolo_anchor_average[:, 1], c='red', s=50);
yoloV3anchors = yolo_anchor_average
yoloV3anchors[:, 0] =yolo_anchor_average[:, 0] /1280 *608
yoloV3anchors[:, 1] =yolo_anchor_average[:, 1] /720 *608
yoloV3anchors = np.rint(yoloV3anchors)
fig, ax = plt.subplots()
for ind in range(n_clusters):
    rectangle= plt.Rectangle((304-yoloV3anchors[ind,0]/2,304-yoloV3anchors[ind,1]/2), yoloV3anchors[ind,0],yoloV3anchors[ind,1] , fc='b',edgecolor='b',fill = None)
    ax.add_patch(rectangle)
ax.set_aspect(1.0)
plt.axis([0,608,0,608])
plt.show()
yoloV3anchors.sort(axis=0)
print("Your custom anchor boxes are {}".format(yoloV3anchors))

F = open("YOLOV3_BDD_Anchors.txt", "w")
F.write("{}".format(yoloV3anchors))
F.close() 