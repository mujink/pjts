import cv2
import numpy as np
import matplotlib.pyplot as plt


def Text_line(image):
    """
    init..
        image : (H,W,C) RGB 포멧의 텍스트 이미지
    return..
        his_vector : Text Historgram.
    """
    image = np.where(image < 1, 0, 1)

    his_vector = []
    for i, img in enumerate (image):
        his_vector.append(np.max(img))
    return his_vector

def Line_edge(Texlin):
    """
    init..
        Texlin : list 텍스트 이미지의 히스토그램 백터
    return..
        edge_index : list type edge index
    """
    edge_index = []
    for i in range(len(Texlin)):
        try:
            tmp =  Texlin[i+1] - Texlin[i] / 1
            edge_index.append(tmp)
        except: # index+1 이 범위를 벗어난 경우. 루프 종료
            break
    edge_index.append(0)
    return edge_index

def edge_index(edge_vector, end=False):
    """
    edge_vector : Line_edge 가 반환한 edge_index <= list
    """
    edge_index = []
    end_edge = []
    for index, value  in enumerate(edge_vector):
        if (value > 0):
            edge_index.append(index)
        elif (value < 0):
            edge_index.append(index)

    # if len(edge_index)%2==1 : # 엣지 길이가 짝수여야함.
        # print("edge_index dim {}".format(len(edge_index)))

    if  end:
        end_edge.append(edge_index[0])
        end_edge.append(edge_index[-1])
        return end_edge
    else:    
        edge_index = np.array(edge_index)
        edge_index = edge_index.reshape(-1,2) # 글자 벡터 당 글자 시작과 끝 2가지 엣지를 쓰기 때문.
        return edge_index


def index_filter(local_index, word):
    """
    local_index : edge_index 의 리턴 값. type = np.array, shape = (-1,2)
    문자 길이, 문자 사이의 거리를 필터링할 기준을 반환.
    letter_dim :문자 길이를 필터링할 기준.
    space_dim  :문자 사이 길이의 기준.
    """
    letter_dim = []
    space_dim = []
    for index, location in enumerate(local_index): # loop (n,2) : n
        letter_dim.append(location[1]-location[0]) # 문자 사이의 거리를 수집
        try:
            space_dim.append(local_index[index+1,0]-local_index[index,1]) # 문자 사이 길이 수집
        except: # index+1 이 범위를 벗어난 경우. 루프 종료
            break
    if word:
        letter_size = int(np.percentile(letter_dim,90)) # 문자 길이중 100 분위의 90%
        return letter_size
    else :
        letter_size = int(np.percentile(letter_dim,70)) # 문자 길이중 100 분위의 60%
        return letter_size
    
def word_split(img, local_index, word = True):
    """
    init..
        img : Text_Coordinate 의 init image
        local_index : list type의 split 위치 
    return..
        image를 word로 잘라서 반환함.
    """
    img_split = []
    local = []
    swich = True
    if word==True:
        letter_size = index_filter(local_index, word)
    else :
        letter_size = index_filter(local_index, word)

    for i, yl in enumerate(local_index):
        if i == 0:
            set = yl[0]

        if word == True:
            a = img[yl[0]-2:yl[1]+2]
            local.append(yl[0]-2)
            local.append(yl[1]+2)
            img_split.append(a)
        else:
            if i > len(local_index)-2:
                Tu = True
            else:
                Tu = (local_index[i+1][1]-local_index[i][0]<=letter_size*1.2)
            # print("index : ", i, yl[1]-yl[0], ">=",letter_size*0.8)
            # print("index : ", i, yl[1]-yl[0], "<=",letter_size+2 )
            # print("index : ", i, (yl[1]-yl[0]> letter_size*0.9),(yl[1]-yl[0]<=letter_size+2))
            # print("index : ", i, Tu, swich)

            if (yl[1]-yl[0]>= letter_size*0.7)and(yl[1]-yl[0]<=letter_size*1.5):
                a = img[yl[0]-1:yl[1]+1]
                # cv2.imshow("image", a)
                # cv2.waitKey(0)
                # cv2.destroyWindow("detect")
                img_split.append(a)
                local.append(yl[0]-1)
                local.append(yl[1]+2)
                swich = True
            elif (Tu)and(swich == True):
                try:
                    # print("너냐?")
                    a = img[local_index[i][0]-1:local_index[i+1][1]+1]
                    img_split.append(a)
                    # cv2.imshow("image", a)
                    # cv2.waitKey(0)
                    # cv2.destroyWindow("detect")
                    local.append(local_index[i][0]-1)
                    local.append(local_index[i+1][1]+1)
                    swich = False
                except:
                    break
            else:
                swich = True
        # a = img[set:set+letter_size+2]
    local = np.array(local)
    local = local.reshape(-1,2)
    return img_split, local

def image_transpos(image):
    timage = []
    for img in image:
        tmp = np.swapaxes(img, 1,0)
        timage.append(tmp)
    return timage


class Text_Coordinate:
    """
    init..
        img : Image for CHW or HWC, Channel is Gray or RGB format 
    return..
        Image의 Xmin, Ymin, Xmax, Ymax를 반환함.
    """

    def __init__(self, img):
        img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        _, img = cv2.threshold(img,235,255,cv2.THRESH_BINARY_INV)
        # cv2.imshow("image", img)
        # cv2.waitKey(0)
        # cv2.destroyWindow("detect")      
        if not isinstance(img, np.ndarray):
            img = np.array(img)

        if img.shape[0]==3:
            img = img.swapaxes(0, 2).swapaxes(0, 1)

        self.img = img


        self.Page_his_vector            =       Text_line(self.img)
        self.w_minmax                   =       Line_edge(self.Page_his_vector)
        self.w_edge_index               =       edge_index(self.w_minmax)
        self.word_image, self.Y_local   =       word_split(self.img, self.w_edge_index)
        init_word                       =       image_transpos(self.word_image)
        instr  =  []
        self.X_local  =  []
        for word_img in init_word:
            self.Word_his_vector        =       Text_line(word_img)
            self.x_minmax               =       Line_edge(self.Word_his_vector)
            self.x_edge_index           =       edge_index(self.x_minmax)
            str_imag, x_local           =       word_split(word_img, self.x_edge_index, word=False)
            # init_str                =       image_transpos(str_imag)
            instr.append(image_transpos(str_imag))
            self.X_local.append(x_local)

    def Word(self):
        return self.word_image, self.Page_his_vector, self.w_minmax

    def bbox(self):
        bboxs = np.zeros((45,40,5), dtype=np.int)
        for word_line, word_local in enumerate(self.Y_local):
            for str_line, str_local in enumerate(self.X_local[word_line]):
                bboxs[word_line,str_line,0] = 1
                bboxs[word_line,str_line,1] = str_local[0]
                bboxs[word_line,str_line,2] = word_local[0]
                bboxs[word_line,str_line,3] = str_local[1]
                bboxs[word_line,str_line,4] = word_local[1]
        
        return bboxs