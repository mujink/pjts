from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
import tensorflow as tf

# img_path = "D:\python\pjt_odo/2주차\Test_image\images/1.jpeg"
# img_path = "D:\python\pjt_odo//2주차/raccoon_model/raccoon/image/raccoon-26.jpg"
img_path = "D:\python\pjt_odo/3주차_히트맵/010.jpg"
print(img_path)
img = image.load_img(img_path, target_size=(224,224))

ximg = image.img_to_array(img)
print(ximg.shape)
x = np.expand_dims(ximg, axis=0)
x = preprocess_input(x)

model = VGG16(weights="imagenet")
model.summary()

preds = model.predict(x)
predicted = decode_predictions(preds, top = 3)
# print(predicted)

african_elephant_output = model.output[:,386]

last_conv_layer = model.get_layer('block5_conv3')

heatmap_model = Model([model.input] , [last_conv_layer.output , model.output])

# grads = K.gradients(african_elephant_output, last_conv_layer.output)[0]
with tf.GradientTape() as tape:
    conv_output, predictions = heatmap_model(x)
    loss = predictions[:, np.argmax(predictions[0])]
    grads = tape.gradient(loss, conv_output)
    pooled_grads = K.mean(grads, axis=(0,1,2))

# iterate = K.function([model.input],
#                      [pooled_grads, last_conv_layer.output[0]])

# pooled_grads_value, conv_layer_output_value = iterate([x])

# for i in range(512):
#     conv_layer_output_value[:,:,i] *= pooled_grads_value[i]

heatmap = np.mean(tf.multiply(pooled_grads,conv_output), axis= -1)
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
plt.matshow(heatmap[0])
plt.show()
# img = cv2.imread(img_path,cv2.IMREAD_COLOR)
# print(img)
heatmap = cv2.resize(heatmap[0], (ximg.shape[1], ximg.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = heatmap * 0.4 + img
cv2.imwrite("test3.jpg", superimposed_img)
cv2.imwrite("test4.jpg", heatmap)

print("무신일인교")