from lbp_score import LBP,LBP_Score
from cyclegan_plates import CycleGAN
import utils
import json
from matplotlib import pyplot as plt
'''
1. 读取文件夹内的模型
2. 通过模型和原始样本，生成新样本，存储在临时文件夹下
3. 计算原始集合生成集的lbpscore，并存储lbpscore
4. 清空临时文件夹，开始下一个循环。
'''
a = utils.get_dir_filelist_by_extension('2018-03-27 09_11_11 Sc/saved_model','h5',with_parent_path=True)
a.sort()
print(a)

source_folder = 'datasets/plates/单独训练集/Sc/trainA'
target_folder = './tmp/translation'
# for model in a:
b = utils.get_dir_filelist_by_extension(source_folder,ext='jpg')

all_lbp_scores = []
k = 1
gan = CycleGAN(dataset_name='/plates/单独训练集/Sc')

for model in a :
    gan.translate(model_path=model,total_num=len(b),output_dir=target_folder)
    ls = LBP_Score(folder_A=source_folder,folder_B=target_folder)
    s = ls.get_lbp_score()
    lbp_score = round(s,4)
    all_lbp_scores.append(lbp_score)
    print('lbp score:',round(s,4))


f = open(gan.output_dir+'/lbpscore.txt',mode='w')
tmp = json.dumps(all_lbp_scores)
f.write(tmp)
f.close()

plt.plot(all_lbp_scores)
plt.show()
plt.savefig(gan.output_dir+'/lbp_score.png')
