import numpy as np
import cv2

def image_preprocess(image, target_size, gt_boxes=None):
    
    ih, iw    = target_size
    h,  w, _  = image.shape

    scale = min(iw/w, ih/h)

    nw, nh  = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    image_paded = np.full(shape=[ih, iw, 3], fill_value=0.0)
    dw, dh = (iw - nw) // 2, (ih - nh) // 2

    image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized

    image_paded = image_paded / 255.
    # print(gt_boxes)
    # print(gt_boxes[:, [0, 2, 4, 6]] * scale + dw)
    # print(gt_boxes[:, [1, 3, 5, 7]] * scale + dh)
    if gt_boxes is None:
        return image_paded
    
    else:
        try:
            gt_boxes[:, [0, 2, 4, 6]] = gt_boxes[:, [0, 2, 4, 6]] * scale + dw
            gt_boxes[:, [1, 3, 5, 7]] = gt_boxes[:, [1, 3, 5, 7]] * scale + dh
        except:
            print(gt_boxes.shape)
            gt_boxes = gt_boxes[np.newaxis,:]
            gt_boxes[:, [0, 2, 4, 6]] = gt_boxes[:, [0, 2, 4, 6]] * scale + dw
            gt_boxes[:, [1, 3, 5, 7]] = gt_boxes[:, [1, 3, 5, 7]] * scale + dh
        return image_paded, gt_boxes