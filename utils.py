import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import NoNorm
import cv2

def get_dir_filelist_by_extension(dir, ext,with_parent_path=False):
    r = os.listdir(dir)
    file_list = []
    for item in r:
        if item.split('.')[-1] == ext:
            if with_parent_path:
                file_list.append(dir + '/' + item)
            else:
                file_list.append(item)
    return file_list
def get_dir_subdir(dir,with_parent_path=False):
    r = os.listdir(dir)
    file_list = []
    for item in r:
        if item.find('.') == -1:
            if with_parent_path:
                file_list.append(dir + '/' + item)
            else:
                file_list.append(item)
    file_list.sort()
    return file_list
def create_new_empty_dir(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)

def fixed_length(num,length):
    a = str(num)
    while len(a) < length:
        a = '0'+a
    return a

def view_pics(pics,cols,titles,output_full_path='tmp.png'):
    #pics应为一维数组,每个为单通道灰度图
    c = cols
    r = int(np.ceil(len(pics)/c))
    print('子图共%d行，%d列'%(r,c))
    fig, axs = plt.subplots(r, c)
    fig.set_size_inches(w=c*1.5, h=r*1.5)
    fig.set_dpi(300)

    # fig.savefig('test2png.png', dpi=100)
    print('画布生成完毕')
    cnt = 0
    gen_imgs = np.array(pics)
    # Rescale images 0 - 1
    gen_imgs =  gen_imgs/255
    print('图像归一化完毕')
    for i in range(r):
        for j in range(c):
            if cnt>= len(pics):break
            t = gen_imgs[cnt]
            axs[i, j].imshow(t, cmap='gray',norm=NoNorm())  #NoNorm去除默认的归一化
            if i==0: axs[i, j].set_title(titles[j])
            if i!=0 and j==0:axs[i, j].set_title(i,loc='left',color='r')
            axs[i, j].axis('off')
            cnt += 1
        print('子图已绘制%d/%d行'%(i,r))
    fig.savefig(output_full_path)
    plt.close()
if __name__ == '__main__':
    d = get_dir_subdir('2018-04-25 10_29_02/predicts',True)
    pics = []
    for subd in d:
        imgs_path = get_dir_filelist_by_extension(subd,'jpg',True)
        for k in imgs_path:
            img = cv2.imread(k, flags=cv2.IMREAD_GRAYSCALE)
            pics.append(img)
        print('%s 处理完成'%subd)

    b = get_dir_filelist_by_extension(d[0],'jpg')
    a = len(b)
    view_pics(pics,a,b)