from cyclegan_plates import CycleGAN
import os
if __name__ == "__main__":
    # 控制显卡可见性
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    cls = ['Cr','In','Pa','PS','RS','Sc']
    # 开始训练
    lambda_gan = 1
    lambda_cycle = 10
    lambda_id = 0.5
    for kind in cls:#_B_plates_new
        gan = CycleGAN('/plates/单独训练集_B_plates_new/'+kind, appendix=kind+'_new_plates_%d%d%d'%(lambda_gan,lambda_cycle,lambda_id),
                       lamb=[lambda_gan,lambda_cycle,lambda_id])#lamb:gan,cycle,id
        gan.train(epochs=2001, batch_size=2, save_interval=100)

    for kind in cls:#_B_plates_new
        gan = CycleGAN('/plates/单独训练集_B_aluminum/'+kind, appendix=kind+'_aluminum_%d%d%d'%(lambda_gan,lambda_cycle,lambda_id),
                       lamb=[lambda_gan,lambda_cycle,lambda_id])#lamb:gan,cycle,id
        gan.train(epochs=2001, batch_size=2, save_interval=100)
