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


def step1(path):
    kinds = ['Cr','In','Pa','PS','RS','Sc']
    sets = ['new_plates','aluminum']
    current_kind = ''
    current_set = ''
    for kind in kinds:
        if kind in path: current_kind = kind;break
    for se in sets:
        if se in path: current_set = se;break
    print('path:',path)
    print(current_set,current_kind)
    # exit()
    gan = CycleGAN('/plates/单独训练集_B_%s/%s' %(current_set,current_kind))

    model_files_list = utils.get_dir_filelist_by_extension(path+'/saved_model', ext='h5',
                                           with_parent_path=True)
    mflist = []
    for mf in model_files_list:
        if 'gAB' in mf:
            mflist.append(mf)
    mflist.sort()
    for model_file in mflist:
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
    utils.view_pics(pics,a,b,output_full_path='./compareresult/'+working_dir.replace('./','').replace(' ','-')+'.png')

if __name__ == "__main__":
    # 控制显卡可见性
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # cls = ['Cr','In','Pa','PS','RS','Sc']
    # cls = ['Cr']
    f = open('list.txt')
    txt = f.read().split('\n')
    f.close()
    utils.create_new_empty_dir('./compareresult')

    # task1: 读取saved_model下面的json模型，以及参数文件。输入一张图像，实现翻译效果，并显示图像 Done
    for t in txt:
        step1(t)

    # task2:拼图
        step2('./tmp.txt')

