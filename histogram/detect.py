import cv2
import numpy as np
import matplotlib.pyplot as plt

class to_text:

    def __init__(self, img, string=False) -> None:
        self.string = string
        self.cont = 0
        img = np.array(img)
        img = cv2.bitwise_not(img)
        self.img = img

    def Text_line(self,image, erode=False):
        """
        이미지의 0축 데이터의 히스토그램을 백터를 추출
        """
        image = np.where(image < 40, 0, 1)
        # if erode:
        #     k = np.ones((3,3),dtype=np.uint8)
        #     image = cv2.erode(image,k,iterations=1)
        his_vector = []
        for i, img in enumerate (image):
            his_vector.append(np.max(img))
        return his_vector

    def Line_edge(self,Texlin):
        """
        히스토그램에서 엣지를 추출
        """
        c = []
        for i in range(len(Texlin)):
            try:
                tmp =  Texlin[i+1] - Texlin[i] / 1
                c.append(tmp)
            except: # index+1 이 범위를 벗어난 경우. 루프 종료
                break
        c.append(0)
        return c

    def edge_index(self, edge_vector):
        """
        엣지 리사이즈 => 글자 백터당 2가지 엣지로 검사하고 바꿈
        """
        edge_index = []
        for index, y  in enumerate(edge_vector):
            if (y == 1):
                edge_index.append(index)
            elif (y == -1):
                edge_index.append(index)
        if len(edge_index)%2==1 : # 엣지 길이가 짝수여야함.
            print("edge_index dim {}".format(len(edge_index)))
        edge_index = np.array(edge_index)
        edge_index = edge_index.reshape(-1,2) # 글자 벡터 당 글자 시작과 끝 2가지 엣지를 쓰기 때문.
        
        return edge_index

    def index_filter(self, local_index):
        """
        문자 길이, 문자 사이의 거리를 필터링할 기준을 반환.
        letter_dim :문자 길이를 필터링할 기준.
        space_dim  :문자 사이 길이의 기준.
        """
        letter_dim = []
        space_dim = []
        for index, location in enumerate(local_index): # loop (n,2) : n
            letter_dim.append(location[1]-location[0]) # 문자 길이를 수집
            try:
                space_dim.append(local_index[index+1,0]-local_index[index,0]) # 문자 사이 길이 수집
            except: # index+1 이 범위를 벗어난 경우. 루프 종료
                break

        letter_size = max(letter_dim) # 문자 길이중 가장 큰 것
        space_size =  max(space_dim)  # 문자 사이 길이 중 가장 큰 것
        return letter_size, space_size

    def Text_split(self, img, y_location, string=False):

        """
        문자, 문자 사이 길이 기준으로 문자를 자름
        """
        dist =[]
        img_split = []
        string_ = {}
        s = []
        letter_size, space_size = self.index_filter(y_location)

        if string==False:
            for i, yl in enumerate(y_location):
                if i == 0:
                    set = yl
                else:
                    set = set+letter_size+space_size
                a = img[yl[0]-2:yl[0]+letter_size+2]
                img_split.append(a)
            img_split = np.array(img_split)
            return img_split

        if string==True:
            string_index = 0 
            h = img.shape[1]
            cont = 0
            for index, yl in enumerate(y_location):
                w = yl[1] - yl[0]
                if index < len(y_location)-1:
                    long_w = y_location[index+1,1] - \
                             y_location[index,0]
                else:
                    long_w = 1000
                if ((h/2)+1 < long_w <= h)and(cont%2==0):
                    try:
                        a = img[y_location[index,0]-1:y_location[index+1,1]+2]
                        string_["{}".format(string_index)] = a
                        string_index += 1
                        cont +=1
                    except:
                        break
                elif (cont%2==1):
                    cont = 0
                elif w < h:
                    a = img[yl[0]:yl[1]+2]
                    string_["{}".format(string_index)] = a
                    string_index += 1
                    cont = 0
                        
            return string_
                  
    def image_transpos(self, trans_img):
        # x,y 축 변환하기
        returns = []
        dictionary  ={}
        if isinstance(trans_img, np.ndarray):
            lenth = trans_img.shape
            if len(lenth)==3:
                for i in range(len(trans_img)):
                    img = trans_img[i]
                    img = np.swapaxes(img, 1,0)
                    returns.append(img)
            elif len(lenth)==2:
                img = np.swapaxes(trans_img, 1,0)
                return img
            returns =np.array(returns)
            return returns

        elif isinstance(trans_img, dict):
            for i in range(len(trans_img)):
                img = trans_img["{}".format(i)]
                img = np.swapaxes(img, 1,0)
                dictionary["{}".format(i)] = img
            return dictionary
    
    def run(self):
        count = 0
        dic = {}
        if self.string == False:
            word_his = self.Text_line(self.img)
            word_line_edge = self.Line_edge(word_his)
            y_location = self.edge_index(word_line_edge)
            word_img = self.Text_split(self.img,y_location,string=False)
            word_split = self.image_transpos(word_img)

            return word_split

        elif self.string == True:
            word_his = self.Text_line(self.img)
            word_line_edge = self.Line_edge(word_his)
            # plt.figure(figsize=(15,4))
            # plt.subplot(2,1,1)
            # plt.plot(word_his)


            # plt.subplot(2,1,2)
            # plt.plot(word_line_edge)
            # plt.show()
            y_location = self.edge_index(word_line_edge)
            word_img = self.Text_split(self.img,y_location,string=False)
            word_split = self.image_transpos(word_img)

            for i in range(len(word_split)):
                string_his = self.Text_line(word_split[i],erode=False)
                string_line_edge = self.Line_edge(string_his)
                string_location = self.edge_index(string_line_edge)
                string_img = self.Text_split(word_split[i],string_location,string=True)
                string_split = self.image_transpos(string_img)

                # dir = "./fig2"             
                # plt.figure(figsize=(15,4))
                # plt.subplot(2,1,1)
                # plt.plot(string_his)


                # plt.subplot(2,1,2)
                # plt.plot(string_line_edge)
                # plt.show()
                # plt.savefig(dir + '/{}.png'.format(asd))

                for j in range(len(string_split)):

                    if string_split["{}".format(j)].shape[1] > string_split["{}".format(j)].shape[0]+1:

                        print("==============================")
                        print(i,j)
                        print(string_split["{}".format(j)].shape)
                        cv2.imshow("img", string_split["{}".format(j)])
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                        string2_split = self.image_transpos(string_split["{}".format(j)])
                        string2_his = self.Text_line(string2_split, erode=False)
                        string2_line_edge = self.Line_edge(string2_his)
                        string2_location = self.edge_index(string2_line_edge)
                        string2_img = self.Text_split(string2_split,string2_location,string=True)
                        string2_trans = self.image_transpos(string2_img)


                        for ky in range(len(string2_trans)):

                            # plt.figure(figsize=(15,4))
                            # plt.subplot(2,1,1)
                            # plt.plot(string2_his)


                            # plt.subplot(2,1,2)
                            # plt.plot(string2_line_edge)
                            # plt.show()
                            tmp = string2_trans["{}".format(ky)]
                            dic["{}".format(count)] = tmp
                            count += 1

                    else :
                        dic["{}".format(count)] = string_split["{}".format(j)]
                        count += 1

                    # print(count)

            return dic


