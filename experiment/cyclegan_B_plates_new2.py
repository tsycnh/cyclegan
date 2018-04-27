from cyclegan_plates import CycleGAN
import os
import cv2
from data_loader import DataLoader
import keras
if __name__ == "__main__":
    # 控制显卡可见性
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # cls = ['Cr','In','Pa','PS','RS','Sc']
    # cls = ['Cr']
    # # 开始训练
    # for kind in cls:#_B_plates_new
    #     gan = CycleGAN('/plates/单独训练集_B_plates_new/'+kind, appendix=kind+'_new_plates_151',lamb=[1,5,1])#lamb:gan,cycle,id
    #     gan.train(epochs=2001, batch_size=2, save_interval=100)
    #
    # for kind in cls:#_B_plates_new
    #     gan = CycleGAN('/plates/单独训练集_B_aluminum/'+kind, appendix=kind+'_new_plates_151',lamb=[1,5,1])#lamb:gan,cycle,id
    #     gan.train(epochs=2001, batch_size=2, save_interval=100)

    # task1: 读取saved_model下面的json模型，以及参数文件。输入一张图像，实现翻译效果，并显示图像
    json_file = open('2018-03-27 09_11_11 Sc/saved_model/g_AB_model.json')
    loaded_model_json = json_file.read()
    json_file.close()

    loaded_model = keras.models.model_from_json(loaded_model_json)
    loaded_model.load_weights('2018-03-27 09_11_11 Sc/saved_model/model_epoch_1000.h5')
    print('model loaded')
    kind = 'Sc'
    dataset_name = '/plates/单独训练集/'+kind
    img_rows = 128
    img_cols = 128
    data_loader = DataLoader(dataset_name=dataset_name,
                                  img_res=(img_rows, img_cols))

    imgs_A = data_loader.load_data(domain='A', batch_size=1)

    fakes_B = loaded_model.predict(imgs_A)
    for i, img in enumerate(fakes_B):
        print('img:', i)
        img = (img + 1) * 127.5
        cv2.imwrite('./tmp/test_translation.jpg')

    print('predict finished')