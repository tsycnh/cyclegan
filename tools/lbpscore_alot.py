from lbp_score import LBP,LBP_Score
from cyclegan_plates import CycleGAN
import utils
'''
1. 读取文件夹内的模型
2. 通过模型和原始样本，生成新样本，存储在临时文件夹下
3. 计算原始集合生成集的lbpscore，并存储lbpscore
4. 清空临时文件夹，开始下一个循环。
'''
a = utils.get_dir_filelist_by_extension('../2018-03-27 09_11_11 Sc/saved_model','h5',with_parent_path=True)
print(a)

# for model in a:
gan = CycleGAN(dataset_name='/plates/单独训练集/Sc')
gan.translate(model_path=a[0])