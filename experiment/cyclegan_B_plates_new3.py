from cyclegan_plates import CycleGAN
import os
import cv2
from data_loader import DataLoader
import keras
import matplotlib.pyplot as plt
import numpy as np
import random
import utils
'''
通过已训练模型，批量生成指定数量的翻译图像。
然后通过生成的翻译图像，按照制定单张图像名，生成一张大图。按顺序显示变化
'''


def step1():
    kind = 'Sc'
    gan = CycleGAN('/plates/单独训练集/' + kind)

    model_files_list = utils.get_dir_filelist_by_extension('2018-03-27 09_11_11 Sc/saved_model', ext='h5',
                                                           with_parent_path=True)
    model_files_list.sort()
    for model_file in model_files_list:
        # np.random.seed(0)
        random.seed(0)
        a = model_file.split('.')[0].split('/')[-1].split('_')[-1]
        gan.translate(model_file, total_num=10, output_dir=gan.output_dir + '/predicts/' + a)

    f = open('./tmp.txt', mode='w')
    f.write(gan.output_dir)
    f.close()
def step2(tmpfile):
    f = open(tmpfile)
    working_dir = f.read()
    f.close()

    d = utils.get_dir_subdir(working_dir+'/predicts',True)
    pics = []
    for subd in d:
        imgs_path = utils.get_dir_filelist_by_extension(subd,'jpg',True)
        for k in imgs_path:
            img = cv2.imread(k, flags=cv2.IMREAD_GRAYSCALE)
            pics.append(img)
        print('%s 处理完成'%subd)

    b = utils.get_dir_filelist_by_extension(d[0],'jpg')
    a = len(b)
    utils.view_pics(pics,a,b)

if __name__ == "__main__":
    # 控制显卡可见性
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # cls = ['Cr','In','Pa','PS','RS','Sc']
    # cls = ['Cr']

    # task1: 读取saved_model下面的json模型，以及参数文件。输入一张图像，实现翻译效果，并显示图像 Done
    step1()

    # task2:拼图
    step2('./tmp.txt')

