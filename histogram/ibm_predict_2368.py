import os
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from image_pipeline import out_text
import cv2
from hanspell import spell_checker

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
print(SCRIPT_PATH)
DEFAULT_MODEL_DIR = os.path.join(SCRIPT_PATH, 'saved-model')
DEFAULT_TEST_DIR = SCRIPT_PATH + '/test_data/test_image'
print(DEFAULT_TEST_DIR)
IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
num_image = 3

model = load_model(os.path.join(DEFAULT_MODEL_DIR, 'mymodel4.h5'))

pred_datas = out_text(DEFAULT_TEST_DIR)
page_lst = []

file_handler = open('hangul_number_dict.picl', 'rb')
hangul_number = pickle.load(file_handler)
file_handler.close()
reverse_dict = dict(map(reversed, hangul_number.items()))

def image_preprocess(image, target_size, gt_boxes=None):
    iw, ih    = target_size, target_size
    w,  h = image.shape

    scale = min(iw/w, ih/h)
    nw, nh  = int(scale * w), int(scale * h)

    image_resized = cv2.resize(image, (nw, nh))
    image_resized = np.array(image_resized).reshape(nh,nw,1)

    image_paded = np.full(shape=[ih, iw, 1], fill_value=255.0)
    dw, dh = (iw - nw) // 2, (ih - nh) // 2

    image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized

    return image_paded
b = []
for idxs, num_image in enumerate(pred_datas):
    test_images = np.empty((len(num_image), IMAGE_WIDTH, IMAGE_HEIGHT, 1))
    for idx, in_line in enumerate(num_image):
        image = cv2.cvtColor(in_line,cv2.COLOR_RGB2GRAY)
        test = image_preprocess(image, IMAGE_WIDTH)
        image = 255 - test.copy()
        # image = np.where(image < 3, 0, image)
        # cv2.imshow("detect_Line", image)
        # cv2.waitKey(0)
        # cv2.destroyWindow("detect")
        test = image / 255.
        test = test.reshape(1, IMAGE_WIDTH, IMAGE_WIDTH,1)
        test_images[idx,:,:,:] = test

    y_predict = model.predict(test_images)
    y_pred = [np.argmax(y) for y in y_predict]
    text_lst = [ reverse_dict[value] for value in y_pred]
    page_lst = "".join(text_lst)
    b.append(page_lst)
print(b[0])

# result = spell_checker.check("일요일그건여러가지뜻을가진날이야요휴일공일안식일주일어쩌면사람의일생도꼭그일요일과같은것인지도몰라요어떤 \
# 사람은일요일을참즐거운휴일로맞이하기도하고또어떤사람은애인과의약속이틀어져서아무것도하는일없이공일로지내기도하고또어떤사람은주일로고스란히교회에다\
# 바치기도하고.......제일요일은공일이었어요그리고요한씨의일요일은주일이엇어요공일과주일그건하늘과땅처럼달라요그러나......잘생각해보면같은점이\
# 하나있어요공일도또주일도둘다제것이아니엇다는점그점만은똑같아요제일요일은헛되이우울하게버려졌어요그리고요한씨의일요일은교회에바쳐졌어요받았던곳으로\
# 다시바쳐졌어요그래요저는그렇게생각해요그리고지금은아무것도안가진저와요한씨가이렇게마주서있는거야요20년만에만난그녀의말이었다양명숙")
# print(result)
# print(result.checked)
# print(y_pred)