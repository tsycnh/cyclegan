from cyclegan_plates import CycleGAN
import os
if __name__ == "__main__":
    # 控制显卡可见性
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    cls = ['Cr','In','Pa','PS','RS','Sc']
    # 开始训练
    for kind in cls:#_B_plates_new
        gan = CycleGAN('/plates/单独训练集/'+kind, appendix=kind+'_new_plates',lamb=[1,5,1])#lamb:gan,cycle,id
        gan.train(epochs=2001, batch_size=2, save_interval=100)
