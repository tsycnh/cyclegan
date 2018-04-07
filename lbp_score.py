# 根据两组图像一一对应，求纹理相似度值，即LBP-score
import cv2
import numpy as np
import matplotlib.pyplot as plt
import utils
class LBP():
    def __init__(self,img):
        # print(img.shape)
        self.image = img
        self.height,self.width = img.shape
        self.lbp_img = np.zeros(shape=[self.height,self.width])
        self.calc_lbp_img()
        self.calc_lbp_hist()
        self.calc_lbp_uniform()
        # print(self.lbp_img)
    def calc_lbp_img(self):
        for row in range(1,self.height-1):
            for col in range(1,self.width-1):
                p = (row,col)
                pattern = self.get_single_point_lbp_pattern(p,self.image)
                self.lbp_img[row,col]=pattern
        a = self.lbp_img[1:self.height-1:,1:self.width-1]
        self.lbp_img=a.astype(dtype=np.uint8)
        # print(a)
    def get_single_point_lbp_pattern(self,center_point,img):
        '''
        v1|v2|v3
        v8|v0|v4
        v7|v6|v5
        '''
        cp = center_point
        v0 = img[cp[0], cp[1]]#center value
        v1 = img[cp[0] - 1, cp[1] - 1]
        v2 = img[cp[0] - 1, cp[1]]
        v3 = img[cp[0] - 1, cp[1] + 1]
        v4 = img[cp[0], cp[1] + 1]
        v5 = img[cp[0] + 1, cp[1] + 1]
        v6 = img[cp[0] + 1, cp[1]]
        v7 = img[cp[0] + 1, cp[1] - 1]
        v8 = img[cp[0], cp[1] - 1]
        value = [v1,v2,v3,v4,v5,v6,v7,v8]
        pattern = ''
        for v in value:
            if v>=v0:
                pattern+='1'
            else:
                pattern+='0'
        return int(pattern,2) # 转换成十进制模式值输出，范围0～255

    def calc_lbp_hist(self):
        h = cv2.calcHist([self.lbp_img],[0],None,[256],[0,256])
        hist = np.histogram(self.lbp_img,bins=256)
        self.hist = h#[1:-1]#ist[0][1:-1]# 去除lbp值为0和255的数据
    def calc_lbp_uniform(self):
        d = [0, 1, 2, 3, 4, 6, 7, 8, 12, 14, 15, 16, 24, 28, 30, 31, 32, 48, 56, 60, 62, 63, 64, 96, 112, 120, 124, 126, 127, 128, 129, 131, 135, 143, 159, 191, 192, 193, 195, 199, 207, 223, 224, 225, 227, 231, 239, 240, 241, 243, 247, 248, 249, 251, 252, 253, 254, 255]

        # 不包含0，255两种模式
        # d = [ 1, 2, 3, 4, 6, 7, 8, 12, 14, 15, 16, 24, 28, 30, 31, 32, 48, 56, 60, 62, 63, 64, 96, 112, 120, 124, 126, 127, 128, 129, 131, 135, 143, 159, 191, 192, 193, 195, 199, 207, 223, 224, 225, 227, 231, 239, 240, 241, 243, 247, 248, 249, 251, 252, 253, 254]
        # h_uniform = np.zeros(shape=[len(d),1])
        h_uniform = []
        for index in d:
            v = self.hist[index,0]
            h_uniform.append(v)

        self.uniform_lbp = np.expand_dims(np.array(h_uniform),1)
    def get_lbp_img(self):
        return self.lbp_img
    def get_lbp_hist(self):
        return self.hist
    def get_uniform_lbp_hist(self):
        return self.uniform_lbp

class LBP_Score():
    def __init__(self,folder_A,folder_B):
        # folder A and folder B should have exactly the same content's file name
        self.folder_A = folder_A
        self.folder_B = folder_B
        self.list_A = utils.get_dir_filelist_by_extension(folder_A,'jpg')
        self.list_B = utils.get_dir_filelist_by_extension(folder_B,'jpg')
        if self.list_A!=self.list_B:
            print('两个文件夹内容不符！')
            exit()
        else:
            self.start()
    def start(self):
        rous = []
        for i in range(len(self.list_A)):
            print('calculating img:',i)
            imgA = cv2.imread(self.folder_A+'/'+self.list_A[i],flags=cv2.IMREAD_GRAYSCALE)
            imgB = cv2.imread(self.folder_B+'/'+self.list_B[i],flags=cv2.IMREAD_GRAYSCALE)
            lA = LBP(imgA)
            lB = LBP(imgB)
            histA = lA.get_uniform_lbp_hist()
            histB = lB.get_uniform_lbp_hist()
            rou = cv2.compareHist(histA, histB, method=cv2.HISTCMP_CORREL)
            rous.append(rou)
            # if i == 10:
            #     break
        r = np.array(rous)
        self.mean = np.mean(r)
    def get_lbp_score(self):
        return self.mean

if __name__ == '__main__':
    ls = LBP_Score('datasets/plates/trainA','translation')
    s = ls.get_lbp_score()
    print('lbp score:',round(s,4))
    pass
    # img1 = cv2.imread('datasets/plates/testA/Cr_4.jpg',flags=cv2.IMREAD_GRAYSCALE)
    # img2 = cv2.imread('datasets/plates/testA/Pa_11.jpg',flags=cv2.IMREAD_GRAYSCALE)
    # l1 = LBP(img1)
    # hist1 = l1.get_lbp_hist()
    # l2 = LBP(img2)
    # hist2 = l2.get_lbp_hist()
    # # plt.plot(hist[0])
    # plt.figure(1)
    # plt.plot(hist1)
    # plt.figure(2)
    #
    # plt.plot(hist2)
    # plt.show()
    # re = cv2.compareHist(hist1,hist2,method=cv2.HISTCMP_BHATTACHARYYA)
    #
    #
    # print(re)# 1.0为完全相似